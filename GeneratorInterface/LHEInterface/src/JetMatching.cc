#include <memory>
#include <vector>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"
#include "GeneratorInterface/LHEInterface/interface/JetInput.h"
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"

namespace lhef {

JetMatching::JetMatching(const edm::ParameterSet &params) :
	jetInput(new JetInput(params)),
	jetClustering(new JetClustering(params))
{
}

JetMatching::~JetMatching()
{
}

std::auto_ptr<JetMatching> JetMatching::create(const edm::ParameterSet &params)
{
	return std::auto_ptr<JetMatching>(new JetMatching(params));
}

double JetMatching::match(const HepMC::GenEvent *partonLevel,
                          const HepMC::GenEvent *finalState) const
{
	JetInput::ParticleVector partons = (*jetInput)(partonLevel);
	JetInput::ParticleVector particles = (*jetInput)(finalState);
	std::vector<JetClustering::Jet> jets = (*jetClustering)(particles);

	std::cout << partons.size() << " partons and "
	          << jets.size() << " jets." << std::endl;

	return 1.0;
}

} // namespace lhef
