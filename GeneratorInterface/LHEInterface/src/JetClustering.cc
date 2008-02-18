#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>
#include <HepMC/SimpleVector.h>

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"

namespace lhef {

class JetClustering::Algorithm {
    public:
	Algorithm(const edm::ParameterSet &params) {}
	virtual ~Algorithm() {}

	virtual std::vector<HepMC::FourVector> operator () (
		const std::vector<HepMC::FourVector> &input) const = 0;
};

namespace {
	class KtAlgorithm : public JetClustering::Algorithm {
	    public:
		KtAlgorithm(const edm::ParameterSet &params);
		~KtAlgorithm() {}

	    private:
		std::vector<HepMC::FourVector> operator () (
			const std::vector<HepMC::FourVector> &input) const;

		fastjet::JetDefinition	jetDefinition;
		double			jetPtMin;
	};
} // anonymous namespace

KtAlgorithm::KtAlgorithm(const edm::ParameterSet &params) :
	JetClustering::Algorithm(params),
	jetDefinition(fastjet::kt_algorithm,
	              params.getParameter<double>("ktRParam"),
	              fastjet::Best),
	jetPtMin(params.getParameter<double>("jetPtMin"))
{
}

std::vector<HepMC::FourVector> KtAlgorithm::operator () (
			const std::vector<HepMC::FourVector> &input) const
{
	std::vector<fastjet::PseudoJet> jfInput;
	jfInput.reserve(input.size());
	for(std::vector<HepMC::FourVector>::const_iterator iter = input.begin();
	    iter != input.end(); ++iter)
		jfInput.push_back(fastjet::PseudoJet(
			iter->px(), iter->py(), iter->pz(), iter->e()));

	fastjet::ClusterSequence sequence(jfInput, jetDefinition);
	std::vector<fastjet::PseudoJet> jets =
				sequence.inclusive_jets(jetPtMin);

	std::vector<HepMC::FourVector> result;
	result.reserve(jets.size());
	for(std::vector<fastjet::PseudoJet>::const_iterator iter = jets.begin();
	    iter != jets.end(); ++iter)
		result.push_back(HepMC::FourVector(
			iter->px(), iter->py(), iter->pz(), iter->E()));

	return result;
}

JetClustering::JetClustering(const edm::ParameterSet &params) :
	partonicFinalState(params.getParameter<bool>("partonicFinalState")),
	excludeResonances(params.getParameter<bool>("excludeResonances")),
	onlySignalProcess(params.getParameter<bool>("onlySignalProcess"))
{
	edm::ParameterSet algoParams =
			params.getParameter<edm::ParameterSet>("algorithm");
	std::string algoName =
			algoParams.getParameter<std::string>("name");

	if (algoName == "KT")
		jetAlgo.reset(new KtAlgorithm(algoParams));
	else
		throw cms::Exception("Configuration")
			<< "JetClustering algorithm \"" << algoName
			<< "\" unknown." << std::endl;

	if (params.exists("ignoreParticleIDs")) {
		ignoreParticleIDs =
			params.getParameter< std::vector<unsigned int> >(
							"ignoreParticleIDs");
		std::sort(ignoreParticleIDs.begin(), ignoreParticleIDs.end());
	}
}

JetClustering::~JetClustering()
{
}

std::vector<HepMC::FourVector> JetClustering::run(
					const HepMC::GenEvent *event) const
{
	std::vector<HepMC::FourVector> input = cluster(event);

	return (*jetAlgo)(input);
}

static bool isParton(int pdgId)
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId > 0 && pdgId < 10) || pdgId == 21;
}

static bool isHadron(int pdgId)
{
	pdgId = (pdgId > 0 ? pdgId : -pdgId) % 10000;
	return (pdgId % 10) > 0 &&
	       ((pdgId > 100 && pdgId < 900) ||
	        (pdgId > 1000 && pdgId < 9000));
}

static unsigned int partIdx(const std::vector<const HepMC::GenParticle*> &p,
                            const HepMC::GenParticle *particle)
{
	std::vector<const HepMC::GenParticle*>::const_iterator pos =
			std::lower_bound(p.begin(), p.end(), particle);
	if (pos == p.end() || *pos != particle)
		throw cms::Exception("CorruptedData")
			<< "HepMC::GenEvent corrupted: Unlisted particles"
			   " in decay tree." << std::endl;

	return pos - p.begin();
}

static void invalidateTree(std::vector<bool> &invalid,
                           const std::vector<const HepMC::GenParticle*> &p,
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

static bool hasPartonChildren(std::vector<bool> &invalid,
                              const std::vector<const HepMC::GenParticle*> &p,
                              const HepMC::GenVertex *v)
{
	if (!v)
		return false;

	for(HepMC::GenVertex::particles_out_const_iterator iter =
					v->particles_out_const_begin();
	    iter != v->particles_out_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx])
			continue;

		if (isParton((*iter)->pdg_id()))
			return true;

		const HepMC::GenVertex *v = (*iter)->end_vertex();
		if (hasPartonChildren(invalid, p, v))
			return true;
	}

	return false;
}

static bool fromSignalVertex(std::vector<bool> &invalid,
                             const std::vector<const HepMC::GenParticle*> &p,
                             const HepMC::GenVertex *v,
                             const HepMC::GenVertex *sig)
{
	if (!v)
		return false;
	if (v == sig)
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx])
			return false;

		const HepMC::GenVertex *v = (*iter)->production_vertex();
		if (fromSignalVertex(invalid, p, v, sig))
			return true;
	}

	return false;
}

static bool fromResonance(std::vector<bool> &invalid,
                          const std::vector<const HepMC::GenParticle*> &p,
                          const HepMC::GenVertex *v,
                          const HepMC::GenVertex *sig)
{
	if (v == sig)
		return true;

	for(HepMC::GenVertex::particles_in_const_iterator iter =
					v->particles_in_const_begin();
	    iter != v->particles_in_const_end(); ++iter) {
		unsigned int idx = partIdx(p, *iter);

		if (invalid[idx] || isParton((*iter)->pdg_id())
		                 || isHadron((*iter)->pdg_id()))
			return false;

		const HepMC::GenVertex *v = (*iter)->production_vertex();
		if (!v)
			continue;

		if (fromSignalVertex(invalid, p, v, sig))
			return true;
	}

	return false;
}

std::vector<HepMC::FourVector> JetClustering::cluster(
					const HepMC::GenEvent *event) const
{
	std::vector<const HepMC::GenParticle*> particles;
	for(HepMC::GenEvent::particle_const_iterator iter = event->particles_begin();
	    iter != event->particles_end(); ++iter)
		particles.push_back(*iter);

	std::sort(particles.begin(), particles.end());
	unsigned int size = particles.size();

	std::vector<bool> selected(size, false);
	std::vector<bool> invalid(size, false);

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
			                            particle->end_vertex())) {
				selected[i] = true;
				invalidateTree(invalid, particles,
				               particle->end_vertex());
			}
		}
		if (onlySignalProcess &&
		    fromSignalVertex(invalid, particles,
		                     particle->production_vertex(),
		                     event->signal_process_vertex()))
			invalid[i] = true;
	}

	std::vector<HepMC::FourVector> result;
	for(unsigned int i = 0; i < size; i++) {
		const HepMC::GenParticle *particle = particles[i];
		if (!selected[i] || invalid[i])
			continue;

		if (excludeResonances &&
		    fromResonance(invalid, particles,
		                  particle->production_vertex(),
		                  event->signal_process_vertex())) {
			invalid[i] = true;
			continue;
		}

		unsigned int id = std::abs(particle->pdg_id());
		std::vector<unsigned int>::const_iterator pos =
			std::lower_bound(ignoreParticleIDs.begin(),
			                 ignoreParticleIDs.end(), id);
		if (pos == ignoreParticleIDs.end() || *pos != id)
			result.push_back(particle->momentum());
	}

	return result;
}

} // namespace lhef
