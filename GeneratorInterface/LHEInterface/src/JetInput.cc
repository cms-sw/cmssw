#include <algorithm>
#include <vector>

#include <Math/GenVector/PxPyPzE4D.h>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/LHEInterface/interface/JetInput.h"

namespace lhef {

JetInput::JetInput() :
	partonicFinalState(false),
	excludeResonances(false),
	onlySignalProcess(false)
{
}

JetInput::JetInput(const edm::ParameterSet &params) :
	partonicFinalState(params.getParameter<bool>("partonicFinalState")),
	excludeResonances(params.getParameter<bool>("excludeResonances")),
	onlySignalProcess(params.getParameter<bool>("onlySignalProcess"))
{
	if (params.exists("ignoreParticleIDs"))
		setIgnoredParticles(
			params.getParameter<std::vector<unsigned int> >(
							"ignoreParticleIDs"));
}

JetInput::~JetInput()
{
}

void JetInput::setIgnoredParticles(
			const std::vector<unsigned int> &particleIDs)
{
	ignoreParticleIDs = particleIDs;
	std::sort(ignoreParticleIDs.begin(), ignoreParticleIDs.end());
}

bool JetInput::isParton(int pdgId)
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 0 && pdgId < 6) || pdgId == 7 ||
	       pdgId == 9 || pdgId == 15 || pdgId == 21;
	// tops are not considered "regular" partons
	// but taus are (since they may hadronize later)
}

bool JetInput::isHadron(int pdgId)
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 100 && pdgId < 900) ||
	       (pdgId > 1000 && pdgId < 9000);
}

bool JetInput::isResonance(int pdgId)
{
	// gauge bosons and tops
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 21 && pdgId <= 39) || pdgId == 6 || pdgId == 8;
}

bool JetInput::isIgnored(int pdgId) const
{
	pdgId = pdgId > 0 ? pdgId : -pdgId;
	std::vector<unsigned int>::const_iterator pos =
			std::lower_bound(ignoreParticleIDs.begin(),
			                 ignoreParticleIDs.end(),
			                 (unsigned int)pdgId);
	return pos != ignoreParticleIDs.end() && *pos == (unsigned int)pdgId;
}

static unsigned int partIdx(const JetInput::ParticleVector &p,
                            const HepMC::GenParticle *particle)
{
	JetInput::ParticleVector::const_iterator pos =
			std::lower_bound(p.begin(), p.end(), particle);
	if (pos == p.end() || *pos != particle)
		throw cms::Exception("CorruptedData")
			<< "HepMC::GenEvent corrupted: Unlisted particles"
			   " in decay tree." << std::endl;

	return pos - p.begin();
}

static void invalidateTree(JetInput::ParticleBitmap &invalid,
                           const JetInput::ParticleVector &p,
                           const HepMC::GenVertex *v)
{
	if (!v)
		return;

	for(HepMC::GenVertex::particles_out_const_iterator iter =
					v->particles_out_const_begin();
	    iter != v->particles_out_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx])
			continue;

		invalid[idx] = true;

		const HepMC::GenVertex *v = (*iter)->end_vertex();
		invalidateTree(invalid, p, v);
	}
}

static int testPartonChildren(JetInput::ParticleBitmap &invalid,
                              const JetInput::ParticleVector &p,
                              const HepMC::GenVertex *v)
{
	if (!v)
		return 0;

	for(HepMC::GenVertex::particles_out_const_iterator iter =
					v->particles_out_const_begin();
	    iter != v->particles_out_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx])
			continue;

		if (JetInput::isParton((*iter)->pdg_id()))
			return 1;
		if (JetInput::isHadron((*iter)->pdg_id()))
			return -1;

		const HepMC::GenVertex *v = (*iter)->end_vertex();
		int result = testPartonChildren(invalid, p, v);
		if (result)
			return result;
	}

	return 0;
}

bool JetInput::hasPartonChildren(ParticleBitmap &invalid,
                                 const ParticleVector &p,
                                 const HepMC::GenParticle *particle) const
{
	return testPartonChildren(invalid, p, particle->end_vertex()) > 0;
}

bool JetInput::fromSignalVertex(ParticleBitmap &invalid,
                                const ParticleVector &p,
                                const HepMC::GenParticle *particle,
                                const HepMC::GenVertex *sig) const
{
	unsigned int idx = partIdx(p, particle);

	if (invalid[idx])
		return false;

	const HepMC::GenVertex *v = particle->production_vertex();
	if (!v)
		return false;
	else if (v == sig)
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter)
		if (fromSignalVertex(invalid, p, *iter, sig))
			return true;

	return false;
}

bool JetInput::fromResonance(ParticleBitmap &invalid,
                             const ParticleVector &p,
                             const HepMC::GenParticle *particle,
                             const HepMC::GenVertex *sig) const
{
	unsigned int idx = partIdx(p, particle);
	int id = particle->pdg_id();

	if (invalid[idx] ||
	    (isResonance(id) && particle->status() == 3))
		return true;

	if (!isIgnored(id) && (isParton(id) || isHadron(id)))
		return false;

	const HepMC::GenVertex *v = particle->production_vertex();
	if (!v)
		return false;
	else if (v == sig)
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter)
		if (fromResonance(invalid, p, *iter, sig))
			return true;

	return false;
}

JetInput::ParticleVector JetInput::operator () (
				const HepMC::GenEvent *event) const
{
	if (!event->signal_process_vertex())
		throw cms::Exception("InvalidHepMCEvent")
			<< "HepMC event is lacking signal vertex."
			<< std::endl;

	ParticleVector particles;
	for(HepMC::GenEvent::particle_const_iterator iter = event->particles_begin();
	    iter != event->particles_end(); ++iter)
		particles.push_back(*iter);

	std::sort(particles.begin(), particles.end());
	unsigned int size = particles.size();

	ParticleBitmap selected(size, false);
	ParticleBitmap invalid(size, false);

	for(unsigned int i = 0; i < size; i++) {
		const HepMC::GenParticle *particle = particles[i];
		if (invalid[i])
			continue;
		if (particle->status() == 1)
			selected[i] = true;

		if (partonicFinalState && isParton(particle->pdg_id())) {
			if (!particle->end_vertex() &&
			    particle->status() != 1)
				// some brokenness in event...
				invalid[i] = true;
			else if (!hasPartonChildren(invalid, particles,
			                            particle)) {
				selected[i] = true;
				invalidateTree(invalid, particles,
				               particle->end_vertex());
			}
		}

		if (onlySignalProcess &&
		    !fromSignalVertex(invalid, particles, particle,
		                      event->signal_process_vertex()))
			invalid[i] = true;
	}

	ParticleVector result;
	for(unsigned int i = 0; i < size; i++) {
		const HepMC::GenParticle *particle = particles[i];
		if (!selected[i] || invalid[i])
			continue;

		if (excludeResonances &&
		    fromResonance(invalid, particles, particle,
		                  event->signal_process_vertex())) {
			invalid[i] = true;
			continue;
		}

		if (isIgnored(particle->pdg_id()))
			continue;

		result.push_back(particle);
	}

	return result;
}

} // namespace lhef
