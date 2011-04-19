#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>

namespace edm {
  EventPrincipal::EventPrincipal(
        boost::shared_ptr<ProductRegistry const> reg,
        ProcessConfiguration const& pc) :
          Base(reg, pc, InEvent),
          aux_(),
          luminosityBlockPrincipal_(),
          unscheduledHandler_(),
          moduleLabelsRunning_(),
          eventSelectionIDs_(),
          branchListIndexes_(),
          branchListIndexToProcessIndex_() {}

  void
  EventPrincipal::clearEventPrincipal() {
    clearPrincipal();
    aux_.reset();
    luminosityBlockPrincipal_.reset();
    unscheduledHandler_.reset();
    moduleLabelsRunning_.clear();
    eventSelectionIDs_.reset();
    branchListIndexes_.reset();
    branchListIndexToProcessIndex_.clear();
  }

  void
  EventPrincipal::fillEventPrincipal(std::auto_ptr<EventAuxiliary> aux,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
        boost::shared_ptr<EventSelectionIDVector> eventSelectionIDs,
        boost::shared_ptr<BranchListIndexes> branchListIndexes,
        boost::shared_ptr<BranchMapper> mapper,
        boost::shared_ptr<DelayedReader> rtrv) {
    fillPrincipal(aux->processHistoryID(), mapper, rtrv);
    aux_.reset(aux.release());
    luminosityBlockPrincipal_ = lbp;
    eventSelectionIDs_ = eventSelectionIDs;
    branchListIndexes_ = branchListIndexes;

    if(luminosityBlockPrincipal_) {
      setProcessHistory(*luminosityBlockPrincipal_);
      aux_->setProcessHistoryID(processHistoryID());
    }

    mapper->processHistoryID() = processHistoryID();
    BranchIDListHelper::fixBranchListIndexes(*branchListIndexes_);

    if(productRegistry().productProduced(InEvent)) {
      // Add index into BranchIDListRegistry for products produced this process
      branchListIndexes_->push_back(productRegistry().producedBranchListIndex());
    }

    // Fill in helper map for Branch to ProductID mapping
    for(BranchListIndexes::const_iterator
        it = branchListIndexes_->begin(),
        itEnd = branchListIndexes_->end();
        it != itEnd; ++it) {
      ProcessIndex pix = it - branchListIndexes_->begin();
      branchListIndexToProcessIndex_.insert(std::make_pair(*it, pix));
    }
    // Fill in the product ID's in the groups.
    for(const_iterator it = this->begin(), itEnd = this->end(); it != itEnd; ++it) {
      (*it)->setProvenance(mapper, branchIDToProductID((*it)->branchDescription().branchID()));
    }
  }

  RunPrincipal const&
  EventPrincipal::runPrincipal() const {
    return luminosityBlockPrincipal().runPrincipal();
  }

  RunPrincipal &
  EventPrincipal::runPrincipal() {
    return luminosityBlockPrincipal().runPrincipal();
  }

  void
  EventPrincipal::put(
        ConstBranchDescription const& bd,
        WrapperHolder const& edp,
        std::auto_ptr<ProductProvenance> productProvenance) {

    assert(bd.produced());
    if(!edp.isValid()) {
      throw Exception(errors::InsertFailure, "Null Pointer")
        << "put: Cannot put because ptr to product is null."
        << "\n";
    }
    branchMapperPtr()->insert(*productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    checkUniquenessAndType(edp, g);
    // Group assumes ownership
    g->putProduct(edp, productProvenance);
  }

  void
  EventPrincipal::putOnRead(
        ConstBranchDescription const& bd,
        void const* product,
        std::auto_ptr<ProductProvenance> productProvenance) {

    assert(!bd.produced());
    branchMapperPtr()->insert(*productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    WrapperHolder const edp(product, g->productData().getInterface());
    checkUniquenessAndType(edp, g);
    // Group assumes ownership
    g->putProduct(edp, productProvenance);
  }

  void
  EventPrincipal::resolveProduct_(Group const& g, bool fillOnDemand) const {
    // Try unscheduled production.
    if(g.onDemand()) {
      if(fillOnDemand) {
        unscheduledFill(g.branchDescription().moduleLabel());
      }
      return;
    }

    if(g.branchDescription().produced()) return; // nothing to do.
    if(g.product()) return; // nothing to do.
    if(g.productUnavailable()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    WrapperHolder edp(store()->getProduct(bk, g.productData().getInterface(), this));

    // Now fix up the Group
    checkUniquenessAndType(edp, &g);
    g.putProduct(edp);
  }

  BranchID
  EventPrincipal::pidToBid(ProductID const& pid) const {
    if(!pid.isValid()) {
      throw Exception(errors::ProductNotFound, "InvalidID")
        << "get by product ID: invalid ProductID supplied\n";
    }
    return productIDToBranchID(pid, BranchIDListRegistry::instance()->data(), *branchListIndexes_);
  }

  ProductID
  EventPrincipal::branchIDToProductID(BranchID const& bid) const {
    if(!bid.isValid()) {
      throw Exception(errors::NotFound, "InvalidID")
        << "branchIDToProductID: invalid BranchID supplied\n";
    }
    typedef BranchIDListHelper::BranchIDToIndexMap BIDToIndexMap;
    typedef BIDToIndexMap::const_iterator Iter;
    typedef std::pair<Iter, Iter> IndexRange;

    BIDToIndexMap const& branchIDToIndexMap = BranchIDListRegistry::instance()->extra().branchIDToIndexMap();
    IndexRange range = branchIDToIndexMap.equal_range(bid);
    for(Iter it = range.first; it != range.second; ++it) {
      BranchListIndex blix = it->second.first;
      std::map<BranchListIndex, ProcessIndex>::const_iterator i = branchListIndexToProcessIndex_.find(blix);
      if(i != branchListIndexToProcessIndex_.end()) {
        ProductIndex productIndex = it->second.second;
        ProcessIndex processIndex = i->second;
        return ProductID(processIndex+1, productIndex+1);
      }
    }
    // cannot throw, because some products may legitimately not have product ID's (e.g. pile-up).
    return ProductID();
  }

  BasicHandle
  EventPrincipal::getByProductID(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    SharedConstGroupPtr const& g = getGroup(bid, true, true);
    if(g.get() == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new Exception(errors::ProductNotFound, "InvalidID"));
      *whyFailed
        << "get by product ID: no product with given id: " << pid << "\n";
      return BasicHandle(whyFailed);
    }

    // Check for case where we tried on demand production and
    // it failed to produce the object
    if(g->onDemand()) {
      boost::shared_ptr<cms::Exception> whyFailed(new Exception(errors::ProductNotFound, "InvalidID"));
      *whyFailed
        << "get by product ID: no product with given id: " << pid << "\n"
        << "onDemand production failed to produce it.\n";
      return BasicHandle(whyFailed);
    }
    return BasicHandle(g->productData());
  }

  WrapperHolder
  EventPrincipal::getIt(ProductID const& pid) const {
    return getByProductID(pid).wrapperHolder();
  }

  Provenance
  EventPrincipal::getProvenance(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    return getProvenance(bid);
  }

  void
  EventPrincipal::setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler) {
    unscheduledHandler_ = iHandler;
  }

  boost::shared_ptr<UnscheduledHandler>
  EventPrincipal::unscheduledHandler() const {
     return unscheduledHandler_;
  }

  EventSelectionIDVector const&
  EventPrincipal::eventSelectionIDs() const {
    return *eventSelectionIDs_;
  }

  BranchListIndexes const&
  EventPrincipal::branchListIndexes() const {
    return *branchListIndexes_;
  }

  bool
  EventPrincipal::unscheduledFill(std::string const& moduleLabel) const {

    // If it is a module already currently running in unscheduled
    // mode, then there is a circular dependency related to which
    // EDProducts modules require and produce.  There is no safe way
    // to recover from this.  Here we check for this problem and throw
    // an exception.
    std::vector<std::string>::const_iterator i =
      find_in_all(moduleLabelsRunning_, moduleLabel);

    if(i != moduleLabelsRunning_.end()) {
      throw Exception(errors::LogicError)
        << "Hit circular dependency while trying to run an unscheduled module.\n"
        << "Current implementation of unscheduled execution cannot always determine\n"
        << "the proper order for module execution.  It is also possible the modules\n"
        << "have a built in circular dependence that will not work with any order.\n"
        << "In the first case, scheduling some or all required modules in paths will help.\n"
        << "In the second case, the modules themselves will have to be fixed.\n";
    }

    moduleLabelsRunning_.push_back(moduleLabel);

    if(unscheduledHandler_) {
      unscheduledHandler_->tryToFill(moduleLabel, *const_cast<EventPrincipal *>(this));
    }
    moduleLabelsRunning_.pop_back();
    return true;
  }

  ProductID
  EventPrincipal::oldToNewProductID_(ProductID const& oldProductID) const {
    BranchID bid = branchMapperPtr()->oldProductIDToBranchID(oldProductID);
    if(!bid.isValid()) return oldProductID;
    return branchIDToProductID(bid);
  }
}
