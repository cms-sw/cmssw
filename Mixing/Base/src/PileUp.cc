#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandPoisson.h"

#include <algorithm>

namespace edm {
  PileUp::PileUp(ParameterSet const& pset,double const averageNumber) :
      type_(pset.getParameter<std::string>("type")),
      minBunch_(pset.getParameter<int>("minBunch")),
      maxBunch_(pset.getParameter<int>("maxBunch")),
      averageNumber_(averageNumber),
      intAverage_(static_cast<int>(averageNumber)),
      poisson_(type_ == "poisson"),
      fixed_(type_ == "fixed"),
      none_(type_ == "none"),
      input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
      poissonDistribution_(0) {

   edm::Service<edm::RandomNumberGenerator> rng;
   if (!rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "PileUp requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   CLHEP::HepRandomEngine& engine = rng->getEngine();

   poissonDistribution_ = new CLHEP::RandPoisson(engine, averageNumber_);

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
      }
      result.push_back(eventVector);
    }
  }
}
