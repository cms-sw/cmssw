#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/interface/ProductDeletedException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>

namespace edm {
  EventPrincipal::EventPrincipal(
        boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<BranchIDListHelper const> branchIDListHelper,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender,
        unsigned int streamIndex) :
    Base(reg, reg->productLookup(InEvent), pc, InEvent, historyAppender),
          aux_(),
          luminosityBlockPrincipal_(),
          branchMapperPtr_(new BranchMapper),
          unscheduledHandler_(),
          moduleLabelsRunning_(),
          eventSelectionIDs_(),
          branchIDListHelper_(branchIDListHelper),
          branchListIndexes_(),
          branchListIndexToProcessIndex_(),
          streamID_(streamIndex){}

  void
  EventPrincipal::clearEventPrincipal() {
    clearPrincipal();
    aux_ = EventAuxiliary();
    luminosityBlockPrincipal_.reset();
    branchMapperPtr_.reset(new BranchMapper);
    unscheduledHandler_.reset();
    moduleLabelsRunning_.clear();
    branchListIndexToProcessIndex_.clear();
  }

  void
  EventPrincipal::fillEventPrincipal(EventAuxiliary const& aux,
        ProcessHistoryRegistry const& processHistoryRegistry,
        EventSelectionIDVector&& eventSelectionIDs,
        BranchListIndexes&& branchListIndexes,
        boost::shared_ptr<BranchMapper> mapper,
        DelayedReader* reader) {
    eventSelectionIDs_ = eventSelectionIDs;
    branchMapperPtr_ = mapper;
    branchListIndexes_ = branchListIndexes;
    if(branchIDListHelper_->hasProducedProducts()) {
      // Add index into BranchIDListRegistry for products produced this process
      branchListIndexes_.push_back(branchIDListHelper_->producedBranchListIndex());
    }
    fillEventPrincipal(aux,processHistoryRegistry,reader);
  }

  void
  EventPrincipal::fillEventPrincipal(EventAuxiliary const& aux,
                                     ProcessHistoryRegistry const& processHistoryRegistry,
                                     DelayedReader* reader) {
    fillPrincipal(aux.processHistoryID(), processHistoryRegistry, reader);
    aux_ = aux;
    aux_.setProcessHistoryID(processHistoryID());
    
    if(branchListIndexes_.empty() and branchIDListHelper_->hasProducedProducts()) {
      // Add index into BranchIDListRegistry for products produced this process
      //  if it hasn't already been filled in by the other fillEventPrincipal or by an earlier call to this function
      branchListIndexes_.push_back(branchIDListHelper_->producedBranchListIndex());
    }

    // Fill in helper map for Branch to ProductID mapping
    ProcessIndex pix = 0;
    for(auto const& blindex : branchListIndexes_) {
      branchListIndexToProcessIndex_.insert(std::make_pair(blindex, pix));
      ++pix;
    }

    // Fill in the product ID's in the product holders.
    for(auto const& prod : *this) {
      if (prod->singleProduct()) {
        prod->setProvenance(branchMapperPtr(), processHistory(), branchIDToProductID(prod->branchDescription().branchID()));
      }
    }
  }

  void
  EventPrincipal::setLuminosityBlockPrincipal(boost::shared_ptr<LuminosityBlockPrincipal> const& lbp) {
    luminosityBlockPrincipal_ = lbp;
  }

  void 
  EventPrincipal::setRunAndLumiNumber(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    assert(run == luminosityBlockPrincipal_->run());
    assert(lumi == luminosityBlockPrincipal_->luminosityBlock());
    EventNumber_t event = aux_.id().event();
    aux_.id() = EventID(run, lumi, event);
  }

  RunPrincipal const&
  EventPrincipal::runPrincipal() const {
    return luminosityBlockPrincipal().runPrincipal();
  }

  void
  EventPrincipal::put(
        BranchDescription const& bd,
        WrapperOwningHolder const& edp,
        ProductProvenance const& productProvenance) {

    // assert commented out for DaqSource.  When DaqSource no longer uses put(), the assert can be restored.
    //assert(produced());
    if(!edp.isValid()) {
      throw Exception(errors::InsertFailure, "Null Pointer")
        << "put: Cannot put because ptr to product is null."
        << "\n";
    }
    branchMapperPtr()->insertIntoSet(productProvenance);
    ProductHolderBase* phb = getExistingProduct(bd.branchID());
    assert(phb);
    checkUniquenessAndType(edp, phb);
    // ProductHolder assumes ownership
    phb->putProduct(edp, productProvenance);
  }

  void
  EventPrincipal::putOnRead(
        BranchDescription const& bd,
        void const* product,
        ProductProvenance const& productProvenance) {

    assert(!bd.produced());
    branchMapperPtr()->insertIntoSet(productProvenance);
    ProductHolderBase* phb = getExistingProduct(bd.branchID());
    assert(phb);
    WrapperOwningHolder const edp(product, phb->productData().getInterface());
    checkUniquenessAndType(edp, phb);
    // ProductHolder assumes ownership
    phb->putProduct(edp, productProvenance);
  }

  void
  EventPrincipal::resolveProduct_(ProductHolderBase const& phb, bool fillOnDemand,
                                  ModuleCallingContext const* mcc) const {
    // Try unscheduled production.
    if(phb.onDemand()) {
      if(fillOnDemand) {
        unscheduledFill(phb.resolvedModuleLabel(),
                        mcc);
      }
      return;
    }

    if(phb.branchDescription().produced()) return; // nothing to do.
    if(phb.product()) return; // nothing to do.
    if(phb.productUnavailable()) return; // nothing to do.
    if(!reader()) return; // nothing to do.

    // must attempt to load from persistent store
    BranchKey const bk = BranchKey(phb.branchDescription());
    WrapperOwningHolder edp(reader()->getProduct(bk, phb.productData().getInterface(), this));

    // Now fix up the ProductHolder
    checkUniquenessAndType(edp, &phb);
    phb.putProduct(edp);
  }

  BranchID
  EventPrincipal::pidToBid(ProductID const& pid) const {
    if(!pid.isValid()) {
      throw Exception(errors::ProductNotFound, "InvalidID")
        << "get by product ID: invalid ProductID supplied\n";
    }
    return productIDToBranchID(pid, branchIDListHelper_->branchIDLists(), branchListIndexes_);
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

    IndexRange range = branchIDListHelper_->branchIDToIndexMap().equal_range(bid);
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

  unsigned int
  EventPrincipal::transitionIndex_() const {
    return streamID_.value();
  }

  static void throwProductDeletedException(ProductID const& pid, edm::EventPrincipal::ConstProductPtr const phb) {
    ProductDeletedException exception;
    exception<<"get by product ID: The product with given id: "<<pid
    <<"\ntype: "<<phb->productType()
    <<"\nproduct instance name: "<<phb->productInstanceName()
    <<"\nprocess name: "<<phb->processName()
    <<"\nwas already deleted. This is a configuration error. Please change the configuration of the module which caused this exception to state it reads this data.";
    throw exception;    
  }
  
  BasicHandle
  EventPrincipal::getByProductID(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    ConstProductPtr const phb = getProductHolder(bid, true, false, nullptr);
    if(phb == nullptr) {
      boost::shared_ptr<cms::Exception> whyFailed(new Exception(errors::ProductNotFound, "InvalidID"));
      *whyFailed
        << "get by product ID: no product with given id: " << pid << "\n";
      return BasicHandle(whyFailed);
    }

    // Was this already deleted?
    if(phb->productWasDeleted()) {
      throwProductDeletedException(pid, phb);
    }
    // Check for case where we tried on demand production and
    // it failed to produce the object
    if(phb->onDemand()) {
      boost::shared_ptr<cms::Exception> whyFailed(new Exception(errors::ProductNotFound, "InvalidID"));
      *whyFailed
        << "get by product ID: no product with given id: " << pid << "\n"
        << "onDemand production failed to produce it.\n";
      return BasicHandle(whyFailed);
    }
    return BasicHandle(phb->productData());
  }

  WrapperHolder
  EventPrincipal::getIt(ProductID const& pid) const {
    return getByProductID(pid).wrapperHolder();
  }

  Provenance
  EventPrincipal::getProvenance(ProductID const& pid, ModuleCallingContext const* mcc) const {
    BranchID bid = pidToBid(pid);
    return getProvenance(bid, mcc);
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
    return eventSelectionIDs_;
  }

  BranchListIndexes const&
  EventPrincipal::branchListIndexes() const {
    return branchListIndexes_;
  }

  bool
  EventPrincipal::unscheduledFill(std::string const& moduleLabel, 
                                  ModuleCallingContext const* mcc) const {

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
        << "The last module on the stack shown above requested data from the\n"
        << "module with label: '" << moduleLabel << "'.\n"
        << "This is illegal because this module is already running (it is in the\n"
        << "stack shown above, it might or might not be asking for data from itself).\n"
        << "More information related to resolving circular dependences can be found here:\n"
        << "https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideUnscheduledExecution#Circular_Dependence_Errors.";
    }

    UnscheduledSentry sentry(&moduleLabelsRunning_, moduleLabel);

    if(unscheduledHandler_) {
      if(mcc == nullptr) {
        throw Exception(errors::LogicError)
          << "EventPrincipal::unscheduledFill, Attempting to run unscheduled production\n"
          << "with a null pointer to the ModuleCalling Context. This should never happen.\n"
          << "Contact a Framework developer";
      }
      unscheduledHandler_->tryToFill(moduleLabel, *const_cast<EventPrincipal*>(this), mcc);
    }
    return true;
  }
}
