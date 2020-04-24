#include "FWCore/Framework/src/PrincipalCache.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

namespace edm {

  PrincipalCache::PrincipalCache() :
    run_(0U),
    lumi_(0U) {
  }

  PrincipalCache::~PrincipalCache() { }

  
  void PrincipalCache::setNumberOfConcurrentPrincipals(PreallocationConfiguration const& iConfig)
  {
    eventPrincipals_.resize(iConfig.numberOfStreams());
  }

  RunPrincipal&
  PrincipalCache::runPrincipal(ProcessHistoryID const& phid, RunNumber_t run) const {
    if (phid != reducedInputProcessHistoryID_ ||
        run != run_ ||
        runPrincipal_.get() == nullptr) {
      throwRunMissing();
    }
    return *runPrincipal_.get();
  }

  std::shared_ptr<RunPrincipal> const&
  PrincipalCache::runPrincipalPtr(ProcessHistoryID const& phid, RunNumber_t run) const {
    if (phid != reducedInputProcessHistoryID_ ||
        run != run_ ||
        runPrincipal_.get() == nullptr) {
      throwRunMissing();
    }
    return runPrincipal_;
  }

  RunPrincipal&
  PrincipalCache::runPrincipal() const {
    if (runPrincipal_.get() == nullptr) {
      throwRunMissing();
    }
    return *runPrincipal_.get();
  }

  std::shared_ptr<RunPrincipal> const&
  PrincipalCache::runPrincipalPtr() const {
    if (runPrincipal_.get() == nullptr) {
      throwRunMissing();
    }
    return runPrincipal_;
  }

  LuminosityBlockPrincipal&
  PrincipalCache::lumiPrincipal(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    if (phid != reducedInputProcessHistoryID_ ||
        run != run_ ||
        lumi != lumi_ ||
        lumiPrincipal_.get() == nullptr) {
      throwLumiMissing();
    }
    return *lumiPrincipal_.get();
  }

  std::shared_ptr<LuminosityBlockPrincipal> const&
  PrincipalCache::lumiPrincipalPtr(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) const {
    if (phid != reducedInputProcessHistoryID_ ||
        run != run_ ||
        lumi != lumi_ ||
        lumiPrincipal_.get() == nullptr) {
      throwLumiMissing();
    }
    return lumiPrincipal_;
  }

  LuminosityBlockPrincipal&
  PrincipalCache::lumiPrincipal() const {
    if (lumiPrincipal_.get() == nullptr) {
      throwLumiMissing();
    }
    return *lumiPrincipal_.get();
  }

  std::shared_ptr<LuminosityBlockPrincipal> const&
  PrincipalCache::lumiPrincipalPtr() const {
    if (lumiPrincipal_.get() == nullptr) {
      throwLumiMissing();
    }
    return lumiPrincipal_;
  }

  void PrincipalCache::merge(std::shared_ptr<RunAuxiliary> aux, std::shared_ptr<ProductRegistry const> reg) {
    if (runPrincipal_.get() == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::merge\n"
        << "Illegal attempt to merge run into cache\n"
        << "There is no run in cache to merge with\n"
        << "Contact a Framework Developer\n";
    }
    if (inputProcessHistoryID_ != aux->processHistoryID()) {
      if (reducedInputProcessHistoryID_ != processHistoryRegistry_->reducedProcessHistoryID(aux->processHistoryID())) {
        throw edm::Exception(edm::errors::LogicError)
          << "PrincipalCache::merge\n"
          << "Illegal attempt to merge run into cache\n"
          << "Reduced ProcessHistoryID inconsistent with the one already in cache\n"
          << "Contact a Framework Developer\n";
      }
      inputProcessHistoryID_ = aux->processHistoryID();
    }
    if (aux->run() != run_) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::merge\n"
        << "Illegal attempt to merge run into cache\n"
        << "Run number inconsistent with run number already in cache\n"
        << "Contact a Framework Developer\n";
    }
    bool runOK = runPrincipal_->adjustToNewProductRegistry(*reg);
    assert(runOK);
    runPrincipal_->mergeAuxiliary(*aux);
  }

  void PrincipalCache::merge(std::shared_ptr<LuminosityBlockAuxiliary> aux, std::shared_ptr<ProductRegistry const> reg) {
    if (lumiPrincipal_.get() == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::merge\n"
        << "Illegal attempt to merge luminosity block into cache\n"
        << "There is no luminosity block in the cache to merge with\n"
        << "Contact a Framework Developer\n";
    }
    if (inputProcessHistoryID_ != aux->processHistoryID()) {
      if (reducedInputProcessHistoryID_ != processHistoryRegistry_->reducedProcessHistoryID(aux->processHistoryID())) {
        throw edm::Exception(edm::errors::LogicError)
          << "PrincipalCache::merge\n"
          << "Illegal attempt to merge run into cache\n"
          << "Reduced ProcessHistoryID inconsistent with the one already in cache\n"
          << "Contact a Framework Developer\n";
      }
      inputProcessHistoryID_ = aux->processHistoryID();
    }
    if (aux->run() != run_ ||
        aux->luminosityBlock() != lumi_) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::merge\n"
        << "Illegal attempt to merge lumi into cache\n"
        << "Run and lumi numbers are inconsistent with the ones already in the cache\n"
        << "Contact a Framework Developer\n";
    }
    bool lumiOK = lumiPrincipal_->adjustToNewProductRegistry(*reg);
    assert(lumiOK);
    lumiPrincipal_->mergeAuxiliary(*aux);
  }

  void PrincipalCache::insert(std::shared_ptr<RunPrincipal> rp) {
    if (runPrincipal_.get() != nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::insert\n"
        << "Illegal attempt to insert run into cache\n"
        << "Contact a Framework Developer\n";
    }
    if (inputProcessHistoryID_ != rp->aux().processHistoryID()) {
      reducedInputProcessHistoryID_ = processHistoryRegistry_->reducedProcessHistoryID(rp->aux().processHistoryID());
      inputProcessHistoryID_ = rp->aux().processHistoryID();
    }
    run_ = rp->run();
    runPrincipal_ = rp; 
  }

  void PrincipalCache::insert(std::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (lumiPrincipal_.get() != nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::insert\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "Contact a Framework Developer\n";
    }
    if (runPrincipal_.get() == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::insert\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "Run is invalid\n"
        << "Contact a Framework Developer\n";
    }
    if (inputProcessHistoryID_ != lbp->aux().processHistoryID()) {
      if (reducedInputProcessHistoryID_ != processHistoryRegistry_->reducedProcessHistoryID(lbp->aux().processHistoryID())) {
        throw edm::Exception(edm::errors::LogicError)
          << "PrincipalCache::insert\n"
          << "Illegal attempt to insert lumi into cache\n"
          << "luminosity block has ProcessHistoryID inconsistent with run\n"
          << "Contact a Framework Developer\n";
      }
      inputProcessHistoryID_ = lbp->aux().processHistoryID();
    }
    if (lbp->run() != run_) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::insert\n"
        << "Illegal attempt to insert lumi into cache\n"
        << "luminosity block inconsistent with run number of run in cache\n"
        << "Contact a Framework Developer\n";
    }
    lumi_ = lbp->luminosityBlock();
    lumiPrincipal_ = lbp; 
  }

  void PrincipalCache::insert(std::shared_ptr<EventPrincipal> ep) {
    unsigned int iStreamIndex = ep->streamID().value();
    assert(iStreamIndex < eventPrincipals_.size());
    eventPrincipals_[iStreamIndex] = ep;
  }

  void PrincipalCache::deleteRun(ProcessHistoryID const& phid, RunNumber_t run) {
    if (runPrincipal_.get() == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::deleteRun\n"
        << "Illegal attempt to delete run from cache\n"
        << "There is no run in cache to delete\n"
        << "Contact a Framework Developer\n";
    }
    if (reducedInputProcessHistoryID_ != phid ||
        run != run_) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::deleteRun\n"
        << "Illegal attempt to delete run from cache\n"
        << "Run number or reduced ProcessHistoryID inconsistent with those in cache\n"
        << "Contact a Framework Developer\n";
    }
    runPrincipal_.reset();
  }

  void PrincipalCache::deleteLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) {
    if (lumiPrincipal_.get() == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::deleteLumi\n"
        << "Illegal attempt to delete luminosity block from cache\n"
        << "There is no luminosity block in the cache to delete\n"
        << "Contact a Framework Developer\n";
    }
    if (reducedInputProcessHistoryID_ != phid ||
        run != run_ ||
        lumi != lumi_) {
      throw edm::Exception(edm::errors::LogicError)
        << "PrincipalCache::deleteLumi\n"
        << "Illegal attempt to delete luminosity block from cache\n"
        << "Run number, lumi numbers, or reduced ProcessHistoryID inconsistent with those in cache\n"
        << "Contact a Framework Developer\n";
    }
    lumiPrincipal_.reset();
  }

  void PrincipalCache::adjustEventsToNewProductRegistry(std::shared_ptr<ProductRegistry const> reg) {
    for(auto &eventPrincipal : eventPrincipals_) {
      if (eventPrincipal) {
        eventPrincipal->adjustIndexesAfterProductRegistryAddition();
        bool eventOK = eventPrincipal->adjustToNewProductRegistry(*reg);
        assert(eventOK);
      }
    }
  }
  
  void PrincipalCache::adjustIndexesAfterProductRegistryAddition() {
    if (runPrincipal_) {
      runPrincipal_->adjustIndexesAfterProductRegistryAddition();
    }
    if (lumiPrincipal_) {
      lumiPrincipal_->adjustIndexesAfterProductRegistryAddition();
    }
  }

  void
  PrincipalCache::throwRunMissing() const {
    throw edm::Exception(edm::errors::LogicError)
      << "PrincipalCache::runPrincipal\n"
      << "Requested a run that is not in the cache (should never happen)\n"
      << "Contact a Framework Developer\n";
  }

  void
  PrincipalCache::throwLumiMissing() const {
    throw edm::Exception(edm::errors::LogicError)
      << "PrincipalCache::lumiPrincipal or PrincipalCache::lumiPrincipalPtr\n"
      << "Requested a luminosity block that is not in the cache (should never happen)\n"
      << "Contact a Framework Developer\n";
  }
}
