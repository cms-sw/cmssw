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
}  // namespace edm
