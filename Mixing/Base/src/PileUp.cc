#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

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
      seed_(pset.getParameter<int>("seed")),
      input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
      eng_(seed_),
      poissonDistribution_(eng_, averageNumber_),
      flatDistribution_(eng_, 0, maxEventsToSkip_ + 1) {
    if (!(poisson_ || fixed_ || none_)) {
      throw cms::Exception("Illegal parameter value","PileUp::PileUp(ParameterSet const& pset)")
        << "'type' parameter (a string) has a value of '" << type_ << "'.\n"
        << "Legal values are 'poisson', 'fixed', or 'none'\n";
    }
    if (maxEventsToSkip_ != 0) {
      int jump = static_cast<int>(flatDistribution_.fire());
      // std::cout << "Initial SKIP: " << jump << std::endl;
      input_->skipEvents(jump);
    }
  }

  void
  PileUp::readPileUp(std::vector<EventPrincipalVector> & result) {
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      EventPrincipalVector eventVector;
      int n = (none_ ? 0 : (poisson_ ? poissonDistribution_.fire() : intAverage_));
      eventVector.reserve(n);
      while (n > 0) {
        EventPrincipalVector oneResult;
        oneResult.reserve(n);
        input_->readMany(n, oneResult);
        // std::cout << "READ: " << oneResult.size() << std::endl;
        std::copy(oneResult.begin(), oneResult.end(), std::back_inserter(eventVector));
        n -= oneResult.size();
        if (n > 0 && maxEventsToSkip_ != 0) {
	  int jump = static_cast<int>(flatDistribution_.fire());
          // std::cout << "SKIP: " << jump << std::endl;
          input_->skipEvents(jump);
        }
      }
      result.push_back(eventVector);
    }
  }
}
