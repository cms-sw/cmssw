#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/VectorInputSource.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  PileUp::PileUp(ParameterSet const& pset) :
      minBunch_(pset.getParameter<int>("minBunch")),
      maxBunch_(pset.getParameter<int>("maxBunch")),
      averageNumber_(pset.getParameter<double>("averageNumber")),
      intAverage_(static_cast<int>(averageNumber_)),
      poisson_(pset.getParameter<bool>("poisson")),
      seed_(pset.getParameter<int>("seed")),
      input_(VectorInputSourceFactory::get()->makeVectorInputSource(pset, InputSourceDescription()).release()),
      eng_(seed_),
      distribution_(eng_, averageNumber_) {
  }

  void
  PileUp::readPileUp(std::vector<std::vector<EventPrincipal *> > & result) {
    std::vector<EventPrincipal *> oneResult;
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      int const n = (poisson_ ? distribution_.fire() : intAverage_);
      input_->readMany(n, oneResult);
      result.push_back(oneResult);
    }
  }
}
