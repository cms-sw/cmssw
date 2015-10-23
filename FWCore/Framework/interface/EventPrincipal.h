#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------

EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Framework/interface/Principal.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {
  class BranchID;
  class BranchIDListHelper;
  class ProductProvenanceRetriever;
  class DelayedReader;
  class EventID;
  class HistoryAppender;
  class LuminosityBlockPrincipal;
  class ModuleCallingContext;
  class ProductID;
  class StreamContext;
  class ThinnedAssociation;
  class ThinnedAssociationsHelper;
  class ProcessHistoryRegistry;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
  public:
    typedef EventAuxiliary Auxiliary;
    typedef Principal Base;

    typedef Base::ConstProductHolderPtr ConstProductHolderPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(
        std::shared_ptr<ProductRegistry const> reg,
        std::shared_ptr<BranchIDListHelper const> branchIDListHelper,
        std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender,
        unsigned int streamIndex = 0);
    ~EventPrincipal() {}

    void fillEventPrincipal(EventAuxiliary const& aux,
        ProcessHistoryRegistry const& processHistoryRegistry,
                            DelayedReader* reader = nullptr);
    void fillEventPrincipal(EventAuxiliary const& aux,
                            ProcessHistoryRegistry const& processHistoryRegistry,
                            EventSelectionIDVector&& eventSelectionIDs,
                            BranchListIndexes&& branchListIndexes);
    //provRetriever is changed via a call to ProductProvenanceRetriever::deepSwap
    void fillEventPrincipal(EventAuxiliary const& aux,
                            ProcessHistoryRegistry const& processHistoryRegistry,
                            EventSelectionIDVector&& eventSelectionIDs,
                            BranchListIndexes&& branchListIndexes,
                            ProductProvenanceRetriever& provRetriever,
                            DelayedReader* reader = nullptr);

    
    void clearEventPrincipal();

    LuminosityBlockPrincipal const& luminosityBlockPrincipal() const {
      return *luminosityBlockPrincipal_;
    }

    LuminosityBlockPrincipal& luminosityBlockPrincipal() {
      return *luminosityBlockPrincipal_;
    }

    bool luminosityBlockPrincipalPtrValid() {
      return (luminosityBlockPrincipal_) ? true : false;
    }

    void setLuminosityBlockPrincipal(std::shared_ptr<LuminosityBlockPrincipal> const& lbp);

    void setRunAndLumiNumber(RunNumber_t run, LuminosityBlockNumber_t lumi);

    EventID const& id() const {
      return aux().id();
    }

    Timestamp const& time() const {
      return aux().time();
    }

    bool isReal() const {
      return aux().isRealData();
    }

    EventAuxiliary::ExperimentType ExperimentType() const {
      return aux().experimentType();
    }

    int bunchCrossing() const {
      return aux().bunchCrossing();
    }

    int storeNumber() const {
      return aux().storeNumber();
    }

    EventAuxiliary const& aux() const {
      return aux_;
    }

    StreamID streamID() const { return streamID_;}

    LuminosityBlockNumber_t luminosityBlock() const {
      return id().luminosityBlock();
    }

    RunNumber_t run() const {
      return id().run();
    }

    RunPrincipal const& runPrincipal() const;

    std::shared_ptr<ProductProvenanceRetriever> productProvenanceRetrieverPtr() const {return provRetrieverPtr_;}

    void setUnscheduledHandler(std::shared_ptr<UnscheduledHandler> iHandler);
    std::shared_ptr<UnscheduledHandler> unscheduledHandler() const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    BranchListIndexes const& branchListIndexes() const;

    Provenance
    getProvenance(ProductID const& pid, ModuleCallingContext const* mcc) const;

    BasicHandle
    getByProductID(ProductID const& oid) const;

    void put(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase> edp,
        ProductProvenance const& productProvenance);

    void putOnRead(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase> edp,
        ProductProvenance const& productProvenance);

    virtual WrapperBase const* getIt(ProductID const& pid) const override;
    virtual WrapperBase const* getThinnedProduct(ProductID const& pid, unsigned int& key) const override;
    virtual void getThinnedProducts(ProductID const& pid,
                                    std::vector<WrapperBase const*>& foundContainers,
                                    std::vector<unsigned int>& keys) const override;

    ProductID branchIDToProductID(BranchID const& bid) const;

    void mergeProvenanceRetrievers(EventPrincipal const& other) {
      provRetrieverPtr_->mergeProvenanceRetrievers(other.productProvenanceRetrieverPtr());
    }

    using Base::getProvenance;
    
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> preModuleDelayedGetSignal_;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> postModuleDelayedGetSignal_;

    
  private:

    BranchID pidToBid(ProductID const& pid) const;

    edm::ThinnedAssociation const* getThinnedAssociation(edm::BranchID const& branchID) const;

    virtual bool unscheduledFill(std::string const& moduleLabel,
                                 SharedResourcesAcquirer* sra,
                                 ModuleCallingContext const* mcc) const override;

    virtual void readFromSource_(ProductHolderBase const& phb, ModuleCallingContext const* mcc) const override;

    virtual unsigned int transitionIndex_() const override;
    
  private:

    class UnscheduledSentry {
    public:
      UnscheduledSentry(std::vector<std::string>* moduleLabelsRunning, std::string const& moduleLabel) :
        moduleLabelsRunning_(moduleLabelsRunning) {
        moduleLabelsRunning_->push_back(moduleLabel);
      }
      ~UnscheduledSentry() {
        moduleLabelsRunning_->pop_back();
      }
    private:
      std::vector<std::string>* moduleLabelsRunning_;
    };

    EventAuxiliary aux_;

    std::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;

    // Pointer to the 'retriever' that will get provenance information from the persistent store.
    std::shared_ptr<ProductProvenanceRetriever> provRetrieverPtr_;

    // Handler for unscheduled modules
    std::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;

    EventSelectionIDVector eventSelectionIDs_;

    std::shared_ptr<BranchIDListHelper const> branchIDListHelper_;
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper_;

    BranchListIndexes branchListIndexes_;

    std::map<BranchListIndex, ProcessIndex> branchListIndexToProcessIndex_;
    
    StreamID streamID_;

  };

  inline
  bool
  isSameEvent(EventPrincipal const& a, EventPrincipal const& b) {
    return isSameEvent(a.aux(), b.aux());
  }
}
#endif

