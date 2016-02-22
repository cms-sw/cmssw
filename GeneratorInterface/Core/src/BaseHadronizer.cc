
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

#include <random>

namespace gen {

const std::vector<std::string> BaseHadronizer::theSharedResources;

BaseHadronizer::BaseHadronizer( edm::ParameterSet const& ps ) :
 randomIndex_(-1) 
{
    if (ps.exists("RandomizedParameters")) {
      std::vector<edm::ParameterSet> randomizedParameters = ps.getParameter<std::vector<edm::ParameterSet> >("RandomizedParameters");
      randomInitWeights_.resize(randomizedParameters.size());
      randomInitConfigDescriptions_.resize(randomizedParameters.size());
      for (unsigned int irand = 0; irand<randomizedParameters.size(); ++irand) {
        randomInitWeights_[irand] = randomizedParameters[irand].getParameter<double>("ConfigWeight");
        if (randomizedParameters[irand].exists("ConfigDescription")) {
          randomInitConfigDescriptions_[irand] = randomizedParameters[irand].getParameter<std::string>("ConfigDescription");
        }
      }
    }  
  
   runInfo().setFilterEfficiency(
      ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
   runInfo().setExternalXSecLO(
      GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSection", -1.)) );
   runInfo().setExternalXSecNLO(
       GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSectionNLO", -1.)) );

}

void BaseHadronizer::randomizeIndex(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine) {
  if (randomInitWeights_.size()>0) {
    //randomly select from a list of provided configuration sets (for parameter scans)
    
    //seeds std 32-bit mersene twister with 50 random integers plus run and lumi section numbers
    //(random integers will be the same for every lumi section in a job)
    constexpr unsigned int nseeds = 50;
    std::array<unsigned int, nseeds+2> seeds;
    for (unsigned int iseed=0; iseed<nseeds; ++iseed) {
      double drandom = rengine->flat();
      unsigned int uirandom = (unsigned int)(std::numeric_limits<unsigned int>::max()*drandom);
      seeds[iseed] = uirandom;
    }
    seeds[nseeds] = lumi.id().run();
    seeds[nseeds+1] = lumi.id().luminosityBlock();
    std::seed_seq seedseq(seeds.begin(),seeds.end());
    std::mt19937 randgen(seedseq);
    std::discrete_distribution<int> randdist(randomInitWeights_.begin(),randomInitWeights_.end());
    
    randomIndex_ = randdist(randgen);
    
  }
}

}
