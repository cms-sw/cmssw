#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  PrincipalCache::PrincipalCache() { }

  PrincipalCache::~PrincipalCache() { }

  RunPrincipal & PrincipalCache::runPrincipal(ProcessHistoryID const& phid, int run) {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    RunIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      RunKey key(iphid->second, run);
      iter = runPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == runPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested a run that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  RunPrincipal const& PrincipalCache::runPrincipal(ProcessHistoryID const& phid, int run) const {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    ConstRunIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      RunKey key(iphid->second, run);
      iter = runPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == runPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::runPrincipal\n"
        << "Requested a run that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  boost::shared_ptr<RunPrincipal> PrincipalCache::runPrincipalPtr(ProcessHistoryID const& phid, int run) {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    RunIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      RunKey key(iphid->second, run);
      iter = runPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == runPrincipals_.end()) {
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

  LuminosityBlockPrincipal & PrincipalCache::lumiPrincipal(ProcessHistoryID const& phid, int run, int lumi) {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    LumiIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      LumiKey key(iphid->second, run, lumi);
      iter = lumiPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == lumiPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested a lumi that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  LuminosityBlockPrincipal const& PrincipalCache::lumiPrincipal(ProcessHistoryID const& phid, int run, int lumi) const {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    ConstLumiIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      LumiKey key(iphid->second, run, lumi);
      iter = lumiPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == lumiPrincipals_.end()) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::lumiPrincipal\n"
        << "Requested a lumi that is not in the cache (should never happen)\n"
        << "Contact a Framework Developer\n";
    }
    return *iter->second.get();
  }

  boost::shared_ptr<LuminosityBlockPrincipal> PrincipalCache::lumiPrincipalPtr(ProcessHistoryID const& phid, int run, int lumi) {
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    LumiIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      LumiKey key(iphid->second, run, lumi);
      iter = lumiPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == lumiPrincipals_.end()) {
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

  bool PrincipalCache::merge(boost::shared_ptr<RunAuxiliary> aux, boost::shared_ptr<ProductRegistry const> reg) {
    ProcessHistoryID phid = aux->processHistoryID();
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    RunIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      int run = aux->run();
      RunKey key(iphid->second, run);
      iter = runPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == runPrincipals_.end()) {
      return false;
    }
    bool runOK = iter->second->adjustToNewProductRegistry(*reg);
    assert(runOK);
    iter->second->mergeAuxiliary(*aux);
    currentRunPrincipal_ = iter->second;
    return true;
  }

  bool PrincipalCache::merge(boost::shared_ptr<LuminosityBlockAuxiliary> aux, boost::shared_ptr<ProductRegistry const> reg) {
    ProcessHistoryID phid = aux->processHistoryID();
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    LumiIterator iter;
    if (iphid != processHistoryIDsMap_.end()) {
      int run = aux->run();
      int lumi = aux->luminosityBlock();
      LumiKey key(iphid->second, run, lumi);
      iter = lumiPrincipals_.find(key);
    }
    if (iphid == processHistoryIDsMap_.end() || iter == lumiPrincipals_.end()) {
      return false;
    }
    bool lumiOK = iter->second->adjustToNewProductRegistry(*reg);
    assert(lumiOK);
    iter->second->mergeAuxiliary(*aux);
    currentLumiPrincipal_ = iter->second;
    return true;
  }

  bool PrincipalCache::insert(boost::shared_ptr<RunPrincipal> rp) {
    ProcessHistoryID phid = rp->aux().processHistoryID();
    int run = rp->run();
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    if (iphid == processHistoryIDsMap_.end()) {
      processHistoryIDsMap_[phid] = processHistoryIDs_.size();
      processHistoryIDs_.push_back(phid);
      iphid = processHistoryIDsMap_.find(phid);
    }
    RunKey key(iphid->second, run);
    assert(runPrincipals_.find(key) == runPrincipals_.end());
    runPrincipals_[key] = rp;
    currentRunPrincipal_ = rp;
    return true;
  }

  bool PrincipalCache::insert(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    ProcessHistoryID phid = lbp->aux().processHistoryID();
    int run = lbp->run();
    int lumi = lbp->luminosityBlock();
    std::map<ProcessHistoryID, int>::const_iterator iphid = processHistoryIDsMap_.find(phid);
    if (iphid == processHistoryIDsMap_.end()) {
      processHistoryIDsMap_[phid] = processHistoryIDs_.size();
      processHistoryIDs_.push_back(phid);
      iphid = processHistoryIDsMap_.find(phid);
    }
    LumiKey key(iphid->second, run, lumi);
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

  void PrincipalCache::deleteRun(ProcessHistoryID const& phid, int run) {
    RunKey key(processHistoryIDsMap_[phid], run);
    RunIterator iter = runPrincipals_.find(key);
    assert(iter != runPrincipals_.end());
    runPrincipals_.erase(iter);
  }

  void PrincipalCache::deleteLumi(ProcessHistoryID const& phid, int run, int lumi) {
    LumiKey key(processHistoryIDsMap_[phid], run, lumi);
    LumiIterator iter = lumiPrincipals_.find(key);
    assert(iter != lumiPrincipals_.end());
    lumiPrincipals_.erase(iter);
  }

  void PrincipalCache::adjustEventToNewProductRegistry(boost::shared_ptr<ProductRegistry const> reg) {
    if (eventPrincipal_) {
      eventPrincipal_->adjustIndexesAfterProductRegistryAddition();
      bool eventOK = eventPrincipal_->adjustToNewProductRegistry(*reg);
      assert(eventOK);
    }
  }
  
  void PrincipalCache::adjustIndexesAfterProductRegistryAddition() {
    for (ConstRunIterator it = runPrincipals_.begin(), itEnd = runPrincipals_.end(); it != itEnd; ++it) {
      it->second->adjustIndexesAfterProductRegistryAddition();
    }
    for (ConstLumiIterator it = lumiPrincipals_.begin(), itEnd = lumiPrincipals_.end(); it != itEnd; ++it) {
      it->second->adjustIndexesAfterProductRegistryAddition();
    }
  }
}
