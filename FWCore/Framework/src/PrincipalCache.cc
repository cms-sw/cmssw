#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/bind.hpp"

namespace edm {

  PrincipalCache::PrincipalCache() { }

  PrincipalCache::~PrincipalCache() { }

  RunPrincipal & PrincipalCache::runPrincipal(int run) {
    RunIterator iter = runPrincipals_.find(run);
    if (iter == runPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested a run that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  RunPrincipal const& PrincipalCache::runPrincipal(int run) const {
    ConstRunIterator iter = runPrincipals_.find(run);
    if (iter == runPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested a run that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  boost::shared_ptr<RunPrincipal> PrincipalCache::runPrincipalPtr(int run) {
    RunIterator iter = runPrincipals_.find(run);
    if (iter == runPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipalPtr\n"
        << "Requested a run that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return iter->second;
  }

  RunPrincipal & PrincipalCache::runPrincipal() {
    if (currentRunPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested current run and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *currentRunPrincipal_.get();
  }

  RunPrincipal const& PrincipalCache::runPrincipal() const {
    if (currentRunPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested current run and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *currentRunPrincipal_.get();
  }

  boost::shared_ptr<RunPrincipal> PrincipalCache::runPrincipalPtr() {
    if (currentRunPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipalPtr\n"
        << "Requested current run and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return currentRunPrincipal_;
  }

  LuminosityBlockPrincipal & PrincipalCache::lumiPrincipal(int run, int lumi) {
    LumiKey key(run, lumi);
    LumiIterator iter = lumiPrincipals_.find(key);
    if (iter == lumiPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested a lumi that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  LuminosityBlockPrincipal const& PrincipalCache::lumiPrincipal(int run, int lumi) const {
    LumiKey key(run, lumi);
    ConstLumiIterator iter = lumiPrincipals_.find(key);
    if (iter == lumiPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested a lumi that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  boost::shared_ptr<LuminosityBlockPrincipal> PrincipalCache::lumiPrincipalPtr(int run, int lumi) {
    LumiKey key(run, lumi);
    LumiIterator iter = lumiPrincipals_.find(key);
    if (iter == lumiPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipalPtr\n"
        << "Requested a lumi that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return iter->second;
  }

  LuminosityBlockPrincipal & PrincipalCache::lumiPrincipal() {
    if (currentLumiPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested current lumi and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *currentLumiPrincipal_.get();
  }

  LuminosityBlockPrincipal const& PrincipalCache::lumiPrincipal() const {
    if (currentLumiPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested current lumi and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *currentLumiPrincipal_.get();
  }

  boost::shared_ptr<LuminosityBlockPrincipal> PrincipalCache::lumiPrincipalPtr() {
    if (currentLumiPrincipal_.get() == 0) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipalPtr\n"
        << "Requested current lumi and it is not initialized (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return currentLumiPrincipal_;
  }

  bool PrincipalCache::merge(boost::shared_ptr<RunAuxiliary> aux) {
    int run = aux->run();
    RunIterator iter = runPrincipals_.find(run); 
    if (iter == runPrincipals_.end()) {
      return false;
    }
    iter->second->mergeAuxiliary(*aux);
    currentRunPrincipal_ = iter->second;
    return true;
  }

  bool PrincipalCache::merge(boost::shared_ptr<LuminosityBlockAuxiliary> aux) {
    int run = aux->run();
    int lumi = aux->luminosityBlock();
    LumiKey key(run, lumi);
    LumiIterator iter = lumiPrincipals_.find(key); 
    if (iter == lumiPrincipals_.end()) {
      return false;
    }
    iter->second->mergeAuxiliary(*aux);
    currentLumiPrincipal_ = iter->second;
    return true;
  }

  bool PrincipalCache::insert(boost::shared_ptr<RunPrincipal> rp) {
    int run = rp->run();
    assert(runPrincipals_.find(run) == runPrincipals_.end());
    runPrincipals_[run] = rp;
    currentRunPrincipal_ = rp;
    return true;
  }

  bool PrincipalCache::insert(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    int run = lbp->run();
    int lumi = lbp->luminosityBlock();
    LumiKey key(run, lumi);
    assert(lumiPrincipals_.find(key) == lumiPrincipals_.end());
    lumiPrincipals_[key] = lbp;
    currentLumiPrincipal_ = lbp;
    return true;
  }

  bool PrincipalCache::noMoreRuns() {
    return runPrincipals_.empty();
  }

  bool PrincipalCache::noMoreLumis() {
    return lumiPrincipals_.empty();
  }

  RunPrincipal const& PrincipalCache::lowestRun() const {
    ConstRunIterator iter = runPrincipals_.begin();
    return *iter->second.get();
  }

  LuminosityBlockPrincipal const& PrincipalCache::lowestLumi() const {
    ConstLumiIterator iter = lumiPrincipals_.begin();
    return *iter->second.get();
  }

  void PrincipalCache::deleteLowestRun() {
    runPrincipals_.erase(runPrincipals_.begin());
  }

  void PrincipalCache::deleteLowestLumi() {
    lumiPrincipals_.erase(lumiPrincipals_.begin());
  }

  void PrincipalCache::deleteRun(int run) {
    runPrincipals_.erase(runPrincipals_.find(run));
  }

  void PrincipalCache::deleteLumi(int run, int lumi) {
    lumiPrincipals_.erase(lumiPrincipals_.find(LumiKey(run, lumi)));
  }

  void PrincipalCache::adjustToNewProductRegistry(ProductRegistry const& reg) {
    eventPrincipal_->adjustToNewProductRegistry(reg);
    for (LumiIterator i = lumiPrincipals_.begin(), iEnd = lumiPrincipals_.end(); i != iEnd; ++i) {
      i->second->adjustToNewProductRegistry(reg);
    }
    for (RunIterator i = runPrincipals_.begin(), iEnd = runPrincipals_.end(); i != iEnd; ++i) {
      i->second->adjustToNewProductRegistry(reg);
    }
  }
}
