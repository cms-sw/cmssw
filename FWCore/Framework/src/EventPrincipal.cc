#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

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
	  history_(),
	  branchListIndexToProcessIndex_() {}

  void
  EventPrincipal::clearEventPrincipal() {
    clearPrincipal();
    aux_.reset();
    luminosityBlockPrincipal_.reset();
    unscheduledHandler_.reset();
    moduleLabelsRunning_.clear();
    history_.reset();
    branchListIndexToProcessIndex_.clear();
  }

  void
  EventPrincipal::fillEventPrincipal(std::auto_ptr<EventAuxiliary> aux,
	boost::shared_ptr<LuminosityBlockPrincipal> lbp,
	boost::shared_ptr<History> history,
	boost::shared_ptr<BranchMapper> mapper,
	boost::shared_ptr<DelayedReader> rtrv) {
    fillPrincipal(history->processHistoryID(), mapper, rtrv);
    aux_.reset(aux.release());
    luminosityBlockPrincipal_ = lbp;
    history_ = history;

    if (productRegistry().productProduced(InEvent)) {
      addToProcessHistory();
    }

    mapper->processHistoryID() = processHistoryID();
    BranchIDListHelper::fixBranchListIndexes(history_->branchListIndexes());

    if (productRegistry().productProduced(InEvent)) {
      // Add index into BranchIDListRegistry for products produced this process
      history_->addBranchListIndexEntry(BranchIDListRegistry::instance()->extra().producedBranchListIndex());
    }

    // Fill in helper map for Branch to ProductID mapping
    for (BranchListIndexes::const_iterator
	 it = history->branchListIndexes().begin(),
	 itEnd = history->branchListIndexes().end();
	 it != itEnd; ++it) {
      ProcessIndex pix = it - history->branchListIndexes().begin();
      branchListIndexToProcessIndex_.insert(std::make_pair(*it, pix));
    }
    // Fill in the product ID's in the groups.
    for (const_iterator it = this->begin(), itEnd = this->end(); it != itEnd; ++it) {
      (*it)->setProductID(branchIDToProductID((*it)->branchDescription().branchID()));
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
	std::auto_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance) {

    assert(bd.produced());
    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
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
	std::auto_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance) {

    assert(!bd.produced());
    branchMapperPtr()->insert(*productProvenance);
    Group *g = getExistingGroup(bd.branchID());
    assert(g);
    checkUniquenessAndType(edp, g);
    // Group assumes ownership
    g->putProduct(edp, productProvenance);
  }

  void
  EventPrincipal::resolveProduct_(Group const& g, bool fillOnDemand) const {
    // Try unscheduled production.
    if (g.onDemand()) {
      if (fillOnDemand) {
        unscheduledFill(g.branchDescription().moduleLabel());
      }
      return;
    }

    if (g.branchDescription().produced()) return; // nothing to do.
    if (g.product()) return; // nothing to do.
    if (g.productUnavailable()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(g.branchDescription());
    std::auto_ptr<EDProduct> edp(store()->getProduct(bk, this));

    // Now fix up the Group
    checkUniquenessAndType(edp, &g);
    g.putProduct(edp);
  }

  void
  EventPrincipal::resolveProvenance_(Group const& g) const {
    g.resolveProvenance(branchMapperPtr());
  }

  BranchID
  EventPrincipal::pidToBid(ProductID const& pid) const {
    if (!pid.isValid()) {
      throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
        << "get by product ID: invalid ProductID supplied\n";
    }
    return productIDToBranchID(pid, BranchIDListRegistry::instance()->data(), history().branchListIndexes());
  }

  ProductID
  EventPrincipal::branchIDToProductID(BranchID const& bid) const {
    if (!bid.isValid()) {
      throw edm::Exception(edm::errors::NotFound,"InvalidID")
        << "branchIDToProductID: invalid BranchID supplied\n";
    }
    typedef BranchIDListHelper::BranchIDToIndexMap BIDToIndexMap;
    typedef BIDToIndexMap::const_iterator Iter;
    typedef std::pair<Iter, Iter> IndexRange;

    BIDToIndexMap const& branchIDToIndexMap = BranchIDListRegistry::instance()->extra().branchIDToIndexMap();   
    IndexRange range = branchIDToIndexMap.equal_range(bid);
    for (Iter it = range.first; it != range.second; ++it) {
      BranchListIndex blix = it->second.first;
      std::map<BranchListIndex, ProcessIndex>::const_iterator i = branchListIndexToProcessIndex_.find(blix);
      if (i != branchListIndexToProcessIndex_.end()) {
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
    SharedConstGroupPtr const& g = getGroup(bid, true, true, true);
    if (g.get() == 0) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound,"InvalidID") );
      *whyFailed
	<< "get by product ID: no product with given id: "<< pid << "\n";
      return BasicHandle(whyFailed);
    }

    // Check for case where we tried on demand production and
    // it failed to produce the object
    if (g->onDemand()) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound,"InvalidID") );
      *whyFailed
	<< "get by product ID: no product with given id: " << pid << "\n"
        << "onDemand production failed to produce it.\n";
      return BasicHandle(whyFailed);
    }
    return BasicHandle(g->product(), g->provenance());
  }

  EDProduct const *
  EventPrincipal::getIt(ProductID const& pid) const {
    return getByProductID(pid).wrapper();
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
  EventPrincipal::eventSelectionIDs() const
  {
    return history_->eventSelectionIDs();
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

    if (i != moduleLabelsRunning_.end()) {
      throw edm::Exception(errors::LogicError)
        << "Hit circular dependency while trying to run an unscheduled module.\n"
        << "Current implementation of unscheduled execution cannot always determine\n"
        << "the proper order for module execution.  It is also possible the modules\n"
        << "have a built in circular dependence that will not work with any order.\n"
        << "In the first case, scheduling some or all required modules in paths will help.\n"
        << "In the second case, the modules themselves will have to be fixed.\n";
    }

    moduleLabelsRunning_.push_back(moduleLabel);

    if (unscheduledHandler_) {
      unscheduledHandler_->tryToFill(moduleLabel, *const_cast<EventPrincipal *>(this));
    }
    moduleLabelsRunning_.pop_back();
    return true;
  }

  ProductID
  EventPrincipal::oldToNewProductID_(ProductID const& oldProductID) const {
    BranchID bid = branchMapperPtr()->oldProductIDToBranchID(oldProductID);
    if (!bid.isValid()) return oldProductID;
    return branchIDToProductID(bid);
  }
}
