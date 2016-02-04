#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include "TRandom.h"
#include "TFile.h"
#include "TH1F.h"

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset, int const minb, int const maxb, double averageNumber, TH1F * const histo, const bool playback) :
    type_(pset.getParameter<std::string>("type")),
    minBunch_(minb),
    maxBunch_(maxb),
    averageNumber_(averageNumber),
    intAverage_(static_cast<int>(averageNumber)),
    histo_(histo),
    histoDistribution_(type_ == "histo"),
    probFunctionDistribution_(type_ == "probFunction"),
    poisson_(type_ == "poisson"),
    fixed_(type_ == "fixed"),
    none_(type_ == "none"),
    input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
    poissonDistribution_(0),
    playback_(playback),
    sequential_(pset.getUntrackedParameter<bool>("sequential", false)),
    seed_(pset.getParameter<edm::ParameterSet>("nbPileupEvents").getUntrackedParameter<int>("seed",1234))
   {

    
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
	<< "PileUp requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    
    CLHEP::HepRandomEngine& engine = rng->getEngine();
    poissonDistribution_ = new CLHEP::RandPoissonQ(engine, averageNumber_);
    
    // Get seed for the case when using user histogram or probability function
    if (histoDistribution_ || probFunctionDistribution_){ 
      gRandom->SetSeed(seed_);
      LogInfo("MixingModule") << " Change seed for " << type_ << " mode. The seed is set to " << seed_;
    } 
     
        
    if (!(histoDistribution_ || probFunctionDistribution_ || poisson_ || fixed_ || none_)) {
      throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
        << "'type' parameter (a string) has a value of '" << type_ << "'.\n"
        << "Legal values are 'poisson', 'fixed', or 'none'\n";
    }
    
  }

  PileUp::~PileUp() {
    delete poissonDistribution_;
  }

  void
  PileUp::readPileUp(std::vector<EventPrincipalVector> & result,std::vector<std::vector<edm::EventID> > &ids) {
            
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      EventPrincipalVector eventVector;
      int n=0;
      
      if (playback_){
	n = ids[i-minBunch_].size();
      } else if (sequential_) {
	// For now, the use case for sequential read reads only one event at a time.
	n = 1;
      } else {
	
	if (none_){
	  n = 0;
	}else if (poisson_){
	  n = poissonDistribution_->fire();
	}else if (fixed_){
	  n = intAverage_;
	}else if (histoDistribution_ || probFunctionDistribution_){
	  double d = histo_->GetRandom();
	  //n = (int) floor(d + 0.5);  // incorrect for bins with integer edges
	  n = int(d);
	}

      }
      eventVector.reserve(n);
      while (n > 0) {
        EventPrincipalVector oneResult;
        oneResult.reserve(n);
	std::vector<edm::EventID> oneResultPlayback;
	oneResultPlayback.reserve(n);
	if (playback_)   {
	  input_->readManySpecified(ids[i-minBunch_],oneResult);  // playback
	} else if (sequential_) {
	  unsigned int file;
	  input_->readManySequential(n, oneResult, file);  // sequential
	} else  {
	  unsigned int file;   //FIXME: need unsigned filenr?
	  input_->readManyRandom(n, oneResult,file);     //no playback
	  for (int j=0;j<(int)oneResult.size();j++){
	    oneResultPlayback.push_back(oneResult[j]->id());
	  }
	  ids[i-minBunch_] = oneResultPlayback;
	}
        LogDebug("readPileup") << "READ: " << oneResult.size();
        std::copy(oneResult.begin(), oneResult.end(), std::back_inserter(eventVector));
	n -= oneResult.size();
      }
      result.push_back(eventVector);
    }
  }

} //namespace edm
