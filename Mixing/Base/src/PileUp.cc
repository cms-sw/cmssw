#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/TripleRand.h"

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset) :
      type_(pset.getParameter<std::string>("type")),
      minBunch_(pset.getParameter<int>("minBunch")),
      maxBunch_(pset.getParameter<int>("maxBunch")),
      averageNumber_(pset.getParameter<double>("averageNumber")),
      intAverage_(static_cast<int>(averageNumber_)),
      poisson_(type_ == "poisson"),
      fixed_(type_ == "fixed"),
      none_(type_ == "none"),
      maxEventsToSkip_(pset.getUntrackedParameter<unsigned int>("maxEventsToSkip", 0)),
      seed_(0),
      input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
      eng_(0),
      poissonDistribution_(0),
      flatDistribution_(0) {

   edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "PileUp requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   seed_ = static_cast<long>(rng->mySeed());
   eng_ = new CLHEP::TripleRand(seed_);
   poissonDistribution_ = new CLHEP::RandPoisson(*eng_, averageNumber_);
   flatDistribution_ = new CLHEP::RandFlat(*eng_, 0, maxEventsToSkip_ + 1);

    if (!(poisson_ || fixed_ || none_)) {
      throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
        << "'type' parameter (a string) has a value of '" << type_ << "'.\n"
        << "Legal values are 'poisson', 'fixed', or 'none'\n";
    }
    if (maxEventsToSkip_ != 0) {
      int jump = static_cast<int>(flatDistribution_->fire());
      LogInfo("PileUp") << "Initial SKIP: " << jump ;
      input_->skipEvents(jump);
    }
  }

  PileUp::~PileUp() {
    delete eng_;
    delete poissonDistribution_;
    delete flatDistribution_;
  }

  void
  PileUp::readPileUp(std::vector<EventPrincipalVector> & result) {
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      EventPrincipalVector eventVector;
      int n = (none_ ? 0 : (poisson_ ? poissonDistribution_->fire() : intAverage_));
      eventVector.reserve(n);
      while (n > 0) {
        EventPrincipalVector oneResult;
        oneResult.reserve(n);
        input_->readMany(n, oneResult);
        LogDebug("readPileup") << "READ: " << oneResult.size();
        std::copy(oneResult.begin(), oneResult.end(), std::back_inserter(eventVector));
        n -= oneResult.size();
        if (n > 0 && maxEventsToSkip_ != 0) {
	  int jump = static_cast<int>(flatDistribution_->fire());
          LogDebug("readPileup") << "SKIP: " << jump ;
          input_->skipEvents(jump);
        }
      }
      result.push_back(eventVector);
    }
  }
}
