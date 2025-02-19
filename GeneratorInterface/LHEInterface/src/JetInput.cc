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
	onlyHardProcess(false),
	excludeResonances(false),
	tausAsJets(false),
	ptMin(0.0)
{
}

JetInput::JetInput(const edm::ParameterSet &params) :
	partonicFinalState(params.getParameter<bool>("partonicFinalState")),
	onlyHardProcess(params.getParameter<bool>("onlyHardProcess")),
	excludeResonances(false),
	tausAsJets(params.getParameter<bool>("tausAsJets")),
	ptMin(0.0)
{
	if (params.exists("ignoreParticleIDs"))
		setIgnoredParticles(
			params.getParameter<std::vector<unsigned int> >(
							"ignoreParticleIDs"));
	if (params.exists("excludedResonances"))
		setExcludedResonances(
			params.getParameter<std::vector<unsigned int> >(
							"excludedResonances"));
	if (params.exists("excludedFromResonances"))
		setExcludedFromResonances(
			params.getParameter<std::vector<unsigned int> >(
						"excludedFromResonances"));
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

void JetInput::setExcludedResonances(
			const std::vector<unsigned int> &particleIDs)
{
	setExcludeResonances(true);
	excludedResonances = particleIDs;
	std::sort(excludedResonances.begin(), excludedResonances.end());
}

void JetInput::setExcludedFromResonances(
			const std::vector<unsigned int> &particleIDs)
{
	excludedFromResonances = particleIDs;
	std::sort(excludedFromResonances.begin(), excludedFromResonances.end());
}

bool JetInput::isParton(int pdgId) const
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 0 && pdgId < 6) || pdgId == 7 ||
	       pdgId == 9 || (tausAsJets && pdgId == 15) || pdgId == 21;
	// tops are not considered "regular" partons
	// but taus eventually are (since they may hadronize later)
}

bool JetInput::isHadron(int pdgId)
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 100 && pdgId < 900) ||
	       (pdgId > 1000 && pdgId < 9000);
}

static inline bool isContained(const std::vector<unsigned int> &list, int id)
{
	unsigned int absId = (unsigned int)(id > 0 ? id : -id);
	std::vector<unsigned int>::const_iterator pos =
			std::lower_bound(list.begin(), list.end(), absId);
	return pos != list.end() && *pos == absId;
}

bool JetInput::isResonance(int pdgId) const
{
	if (excludedResonances.empty()) {
		// gauge bosons and tops
		pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
		return (pdgId > 21 && pdgId <= 39) || pdgId == 6 || pdgId == 8;
	}

	return isContained(excludedResonances, pdgId);
}

bool JetInput::isIgnored(int pdgId) const
{
	return isContained(ignoreParticleIDs, pdgId);
}

bool JetInput::isExcludedFromResonances(int pdgId) const
{
	if (excludedFromResonances.empty())
		return true;

	return isContained(excludedFromResonances, pdgId);
}

bool JetInput::isHardProcess(const HepMC::GenVertex *vertex,
                             const VertexVector &hardProcess) const
{
	std::vector<const HepMC::GenVertex*>::const_iterator pos =
			std::lower_bound(hardProcess.begin(),
			                 hardProcess.end(), vertex);
	return pos != hardProcess.end() && *pos == vertex;
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

int JetInput::testPartonChildren(JetInput::ParticleBitmap &invalid,
                                 const JetInput::ParticleVector &p,
                                 const HepMC::GenVertex *v) const
{
	if (!v)
		return 0;

	for(HepMC::GenVertex::particles_out_const_iterator iter =
					v->particles_out_const_begin();
	    iter != v->particles_out_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx])
			continue;

		if (isParton((*iter)->pdg_id()))
			return 1;
		if (isHadron((*iter)->pdg_id()))
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

bool JetInput::fromHardProcess(ParticleBitmap &invalid,
                               const ParticleVector &p,
                               const HepMC::GenParticle *particle,
                               const VertexVector &hardProcess) const
{
	unsigned int idx = partIdx(p, particle);

	if (invalid[idx])
		return false;

	const HepMC::GenVertex *v = particle->production_vertex();
	if (!v)
		return false;
	else if (isHardProcess(v, hardProcess))
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter)
		if (fromHardProcess(invalid, p, *iter, hardProcess))
			return true;

	return false;
}

bool JetInput::fromSignalVertex(ParticleBitmap &invalid,
                                const ParticleVector &p,
                                const HepMC::GenParticle *particle,
                                const HepMC::GenVertex *signalVertex) const
{
	unsigned int idx = partIdx(p, particle);

	if (invalid[idx])
		return false;

	const HepMC::GenVertex *v = particle->production_vertex();
	if (!v)
		return false;
	else if (v == signalVertex)
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter)
		if (fromSignalVertex(invalid, p, *iter, signalVertex))
			return true;

	return false;
}

JetInput::ResonanceState
JetInput::fromResonance(ParticleBitmap &invalid,
                        const ParticleVector &p,
                        const HepMC::GenParticle *particle,
                        const HepMC::GenVertex *signalVertex,
                        const VertexVector &hardProcess) const
{
	unsigned int idx = partIdx(p, particle);
	int id = particle->pdg_id();

	if (invalid[idx])
		return kIndirect;

	if (particle->status() == 3 && isResonance(id))
		return kDirect;

	if (!isIgnored(id) && excludedFromResonances.empty() && isParton(id))
		return kNo;

	const HepMC::GenVertex *v = particle->production_vertex();
	if (!v)
		return kNo;
	else if (v == signalVertex && excludedResonances.empty())
		return kIndirect;
	else if (v == signalVertex || isHardProcess(v, hardProcess))
		return kNo;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter) {
		ResonanceState result =
			fromResonance(invalid, p, *iter,
			              signalVertex, hardProcess);
		switch(result) {
		    case kNo:
			break;
		    case kDirect:
			if ((*iter)->pdg_id() == id)
				return kDirect;
			if (!isExcludedFromResonances(id))
				break;
		    case kIndirect:
			return kIndirect;
		}
	}

	return kNo;
}

JetInput::ParticleVector JetInput::operator () (
				const HepMC::GenEvent *event) const
{
	if (!event->signal_process_vertex())
		throw cms::Exception("InvalidHepMCEvent")
			<< "HepMC event is lacking signal vertex."
			<< std::endl;

	VertexVector hardProcess, toLookAt;
	std::pair<HepMC::GenParticle*,HepMC::GenParticle*> beamParticles =
						event->beam_particles();
	toLookAt.push_back(event->signal_process_vertex());
	while(!toLookAt.empty()) {
		std::vector<const HepMC::GenVertex*> next;
		for(std::vector<const HepMC::GenVertex*>::const_iterator v =
			toLookAt.begin(); v != toLookAt.end(); ++v) {
			if (!*v || isHardProcess(*v, hardProcess))
				continue;

			bool veto = false;
			for(HepMC::GenVertex::particles_in_const_iterator iter =
					(*v)->particles_in_const_begin();
			    iter != (*v)->particles_in_const_end(); ++iter) {
				if (*iter == beamParticles.first ||
				    *iter == beamParticles.second) {
					veto = true;
					break;
				}
			}
			if (veto)
				continue;

			hardProcess.push_back(*v);
			std::sort(hardProcess.begin(), hardProcess.end());

			for(HepMC::GenVertex::particles_in_const_iterator iter =
					(*v)->particles_in_const_begin();
			    iter != (*v)->particles_in_const_end(); ++iter) {
				const HepMC::GenVertex *pv =
						(*iter)->production_vertex();
				if (pv)
					next.push_back(pv);
			}
		}

		toLookAt = next;
		std::sort(toLookAt.begin(), toLookAt.end());
	}

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

		if (onlyHardProcess &&
		    !fromHardProcess(invalid, particles,
		                     particle, hardProcess) &&
		    !isHardProcess(particle->end_vertex(), hardProcess))
			invalid[i] = true;
	}

	ParticleVector result;
	for(unsigned int i = 0; i < size; i++) {
		const HepMC::GenParticle *particle = particles[i];
		if (!selected[i] || invalid[i])
			continue;

		if (excludeResonances &&
		    fromResonance(invalid, particles, particle,
		                  event->signal_process_vertex(),
		                  hardProcess)) {
			invalid[i] = true;
			continue;
		}

		if (isIgnored(particle->pdg_id()))
			continue;

		if (particle->momentum().perp() >= ptMin)
			result.push_back(particle);
	}

	return result;
}

} // namespace lhef
