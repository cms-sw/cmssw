#include "FWCore/Framework/interface/PrincipalCache.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"

#include <cassert>

namespace edm {

  PrincipalCache::PrincipalCache() {}

  PrincipalCache::~PrincipalCache() {}

  std::shared_ptr<RunPrincipal> PrincipalCache::getAvailableRunPrincipalPtr() { return runHolder_.tryToGet(); }

  std::shared_ptr<LuminosityBlockPrincipal> PrincipalCache::getAvailableLumiPrincipalPtr() {
    return lumiHolder_.tryToGet();
  }

  void PrincipalCache::setNumberOfConcurrentPrincipals(PreallocationConfiguration const& iConfig) {
    eventPrincipals_.resize(iConfig.numberOfStreams());
  }

  void PrincipalCache::insert(std::unique_ptr<ProcessBlockPrincipal> pb) { processBlockPrincipal_ = std::move(pb); }

  void PrincipalCache::insertForInput(std::unique_ptr<ProcessBlockPrincipal> pb) {
    inputProcessBlockPrincipal_ = std::move(pb);
  }

  void PrincipalCache::insert(std::unique_ptr<RunPrincipal> rp) { runHolder_.add(std::move(rp)); }

  void PrincipalCache::insert(std::unique_ptr<LuminosityBlockPrincipal> lbp) { lumiHolder_.add(std::move(lbp)); }

  void PrincipalCache::insert(std::shared_ptr<EventPrincipal> ep) {
    unsigned int iStreamIndex = ep->streamID().value();
    assert(iStreamIndex < eventPrincipals_.size());
    eventPrincipals_[iStreamIndex] = ep;
  }

  void PrincipalCache::adjustEventsToNewProductRegistry(std::shared_ptr<ProductRegistry const> reg) {
    for (auto& eventPrincipal : eventPrincipals_) {
      if (eventPrincipal) {
        eventPrincipal->adjustIndexesAfterProductRegistryAddition();
        bool eventOK = eventPrincipal->adjustToNewProductRegistry(*reg);
        assert(eventOK);
      }
    }
  }

  void PrincipalCache::adjustIndexesAfterProductRegistryAddition() {
    //Need to temporarily hold all the runs to clear out the runHolder_
    std::vector<std::shared_ptr<RunPrincipal>> tempRunPrincipals;
    while (auto p = runHolder_.tryToGet()) {
      p->adjustIndexesAfterProductRegistryAddition();
      tempRunPrincipals.emplace_back(std::move(p));
    }
    //Need to temporarily hold all the lumis to clear out the lumiHolder_
    std::vector<std::shared_ptr<LuminosityBlockPrincipal>> tempLumiPrincipals;
    while (auto p = lumiHolder_.tryToGet()) {
      p->adjustIndexesAfterProductRegistryAddition();
      tempLumiPrincipals.emplace_back(std::move(p));
    }
  }

}  // namespace edm
