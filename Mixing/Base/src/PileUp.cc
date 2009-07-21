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

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset, int const minb, int const maxb, double averageNumber, const bool playback) :
    type_(pset.getParameter<std::string>("type")),
    minBunch_(minb),
    maxBunch_(maxb),
    averageNumber_(averageNumber),
    intAverage_(static_cast<int>(averageNumber)),
    poisson_(type_ == "poisson"),
    fixed_(type_ == "fixed"),
    none_(type_ == "none"),
    input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
    poissonDistribution_(0),
    playback_(playback),
    sequential_(pset.getUntrackedParameter<bool>("sequential", false))
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

    if (!(poisson_ || fixed_ || none_)) {
      throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
        << "'type' parameter (a string) has a value of '" << type_ << "'.\n"
        << "Legal values are 'poisson', 'fixed', or 'none'\n";
    }
  }

  PileUp::~PileUp() {
    delete poissonDistribution_;
  }

  void
  PileUp::readPileUp(std::vector<EventPrincipalVector> & result,std::vector<edm::EventID> &ids, std::vector<int> &fileNrs,std::vector<unsigned int> & nrEvents) {
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      EventPrincipalVector eventVector;
      int n;
      
      if (playback_){
	n=nrEvents[i-minBunch_];
      } else if (sequential_) {
	// For now, the use case for sequential read reads only one event at a time.
	n = 1;
      } else {
	n = (none_ ? 0 : (poisson_ ? poissonDistribution_->fire() : intAverage_));
	nrEvents[i-minBunch_]=n;
      }
      eventVector.reserve(n);
      while (n > 0) {
        EventPrincipalVector oneResult;
        oneResult.reserve(n);
	if (playback_)   {
	  input_->readMany(n, oneResult,ids[i-minBunch_],fileNrs[i-minBunch_]);  // playback
	} else if (sequential_) {
	  input_->readMany(n, oneResult);  // sequential
	} else  {
	  unsigned int file;   //FIXME: need unsigned filenr?
	  input_->readManyRandom(n, oneResult,file);     //no playback
          ids[i-minBunch_]=oneResult[0]->id(); 
	  fileNrs[i-minBunch_]=file;
	}
        LogDebug("readPileup") << "READ: " << oneResult.size();
        std::copy(oneResult.begin(), oneResult.end(), std::back_inserter(eventVector));
	n -= oneResult.size();
      }
      result.push_back(eventVector);
    }
  }
} //namespace edm
