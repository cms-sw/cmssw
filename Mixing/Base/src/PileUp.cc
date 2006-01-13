#include "Mixing/Base/interface/PileUp.h"
#include "FWCore/Framework/interface/Event.h"
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
      distribution_(eng_, averageNumber_),
      md_() {
    md_.pid = pset.id();
    md_.moduleName_ = pset.getUntrackedParameter<std::string>("@module_type");
    md_.moduleLabel_ = pset.getUntrackedParameter<std::string>("@module_label");
//#warning process name is hard coded, for now.  Fix this.
    md_.processName_ = "PILEUP";
//#warning version and pass are hardcoded
    md_.versionNumber_ = 1;
    md_.pass = 1;
  }

  void
  PileUp::readPileUp(std::vector<std::vector<Event *> > & result) {
    // WARNING::  This leaks memory, as EventPrincipal is never reclaimed.
    // This needs to be fixed.
    for (int i = minBunch_; i <= maxBunch_; ++i) {
      std::vector<EventPrincipal *> oneResult;
      std::vector<Event *> eventVector;
      int const n = (poisson_ ? distribution_.fire() : intAverage_);
      input_->readMany(n, oneResult);
      std::vector<EventPrincipal *>::const_iterator it = oneResult.begin();
      for (; it != oneResult.end(); ++it) {
        Event *e = new Event(**it, md_);
        eventVector.push_back(e);
      }
      result.push_back(eventVector);
    }
  }
}
