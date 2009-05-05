

// $Id: PrincipalCache.cc,v 1.3 2009/04/26 16:00:34 chrjones Exp $

#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

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

  bool PrincipalCache::insert(boost::shared_ptr<RunPrincipal> rp) {
    int run = rp->run();
    RunIterator iter = runPrincipals_.find(run); 
    if (iter == runPrincipals_.end()) {
      runPrincipals_[run] = rp;
      currentRunPrincipal_ = rp;
      return true;
    }
    //the new RunPrincipal has the structure which matches the updated ProductRegistry
    // we must swap the objects because the pointer to the object is used elsewhere
    iter->second->swap(*rp);
    iter->second->mergeRun(rp);
    currentRunPrincipal_ = iter->second;
    
    return true;
  }

  bool PrincipalCache::insert(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    int run = lbp->run();
    int lumi = lbp->luminosityBlock();
    LumiKey key(run, lumi);
    LumiIterator iter = lumiPrincipals_.find(key); 
    if (iter == lumiPrincipals_.end()) {
      lumiPrincipals_[key] = lbp;
      currentLumiPrincipal_ = lbp;
      return true;
    }

    //the new LuminosityBlockPrincipal has the structure which matches the updated ProductRegistry
    // we must swap the objects because the pointer to the object is used elsewhere
    iter->second->swap(*lbp);
    iter->second->mergeLuminosityBlock(lbp);
    currentLumiPrincipal_ = iter->second;
    
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
}
