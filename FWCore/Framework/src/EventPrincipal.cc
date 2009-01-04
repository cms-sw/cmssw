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
  EventPrincipal::EventPrincipal(EventAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	boost::shared_ptr<History> history,
	boost::shared_ptr<BranchMapper> mapper,
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, history->processHistoryID(), mapper, rtrv),
	  aux_(aux),
	  luminosityBlockPrincipal_(),
	  unscheduledHandler_(),
	  moduleLabelsRunning_(),
	  history_(history),
	  branchListIndexToProcessIndex_() {
	    if (reg->productProduced(InEvent)) {
	      addToProcessHistory();
	      // Add index into BranchIDListRegistry for products produced this process
	      history_->addBranchListIndexEntry(BranchIDListRegistry::instance()->extra().producedBranchListIndex());
	    }
	    mapper->processHistoryID() = processHistoryID();
	    BranchIDListHelper::fixBranchListIndexes(history_->branchListIndexes());
	    // Fill in helper map for Branch to ProductID mapping
	    for (BranchListIndexes::const_iterator
		 it = history->branchListIndexes().begin(),
		 itEnd = history->branchListIndexes().end();
		 it != itEnd; ++it) {
	      ProcessIndex pix = it - history->branchListIndexes().begin();
	      branchListIndexToProcessIndex_.insert(std::make_pair(*it, pix));
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
  EventPrincipal::addOnDemandGroup(ConstBranchDescription const& desc) {
    std::auto_ptr<Group> g(new Group(desc, branchIDToProductID(desc.branchID()), true));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {
    Group const* group = getExistingGroup(*g);
    if (group != 0) {
      if(!group->onDemand()) {
        ConstBranchDescription const& bd = group->productDescription();
	throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "addGroup_: Problem found while adding product provenance, "
	  << "product already exists for ("
	  << bd.friendlyClassName() << ","
	  << bd.moduleLabel() << ","
	  << bd.productInstanceName() << ","
	  << bd.processName()
	  << ")\n";
      }
      replaceGroup(g);
    } else {
      addGroup_(g);
    }
  }

  void
  EventPrincipal::addGroup(ConstBranchDescription const& bd) {
    std::auto_ptr<Group> g(new Group(bd, branchIDToProductID(bd.branchID())));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addGroup(std::auto_ptr<EDProduct> prod,
	 ConstBranchDescription const& bd,
	 std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(prod, bd, branchIDToProductID(bd.branchID()), productProvenance));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addGroup(ConstBranchDescription const& bd,
	 std::auto_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(bd, branchIDToProductID(bd.branchID()), productProvenance));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addGroup(std::auto_ptr<EDProduct> prod,
	 ConstBranchDescription const& bd,
	 boost::shared_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(prod, bd, branchIDToProductID(bd.branchID()), productProvenance));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addGroup(ConstBranchDescription const& bd,
	 boost::shared_ptr<ProductProvenance> productProvenance) {
    std::auto_ptr<Group> g(new Group(bd, branchIDToProductID(bd.branchID()), productProvenance));
    addOrReplaceGroup(g);
  }

  void 
  EventPrincipal::put(std::auto_ptr<EDProduct> edp,
		ConstBranchDescription const& bd,
		std::auto_ptr<ProductProvenance> productProvenance) {

    if (edp.get() == 0) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Pointer")
	<< "put: Cannot put because auto_ptr to product is null."
	<< "\n";
    }
    ProductID pid = branchIDToProductID(bd.branchID());
    // Group assumes ownership
    if (!pid.isValid()) {
      throw edm::Exception(edm::errors::InsertFailure,"Null Product ID")
	<< "put: Cannot put product with null Product ID."
	<< "\n";
    }
    branchMapperPtr()->insert(*productProvenance);
    this->addGroup(edp, bd, productProvenance);
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
    throw edm::Exception(edm::errors::NotFound,"Bad BranchID")
      << "branchIDToProductID: productID cannot be determined from BranchID\n";
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
    return branchIDToProductID(branchMapperPtr()->oldProductIDToBranchID(oldProductID));
  }
}
