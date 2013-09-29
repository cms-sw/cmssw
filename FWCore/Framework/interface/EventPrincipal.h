#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------

EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/Principal.h"

#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {
  class BranchIDListHelper;
  class ProductProvenanceRetriever;
  class DelayedReader;
  class EventID;
  class HistoryAppender;
  class LuminosityBlockPrincipal;
  class ModuleCallingContext;
  class ProcessHistoryRegistry;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
  public:
    typedef EventAuxiliary Auxiliary;
    typedef Principal Base;

    typedef Base::ConstProductPtr ConstProductPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(
        boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<BranchIDListHelper const> branchIDListHelper,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender,
        unsigned int streamIndex = 0);
    ~EventPrincipal() {}

    void fillEventPrincipal(EventAuxiliary const& aux,
        ProcessHistoryRegistry const& processHistoryRegistry,
                            DelayedReader* reader = 0);

    void fillEventPrincipal(EventAuxiliary const& aux,
                            ProcessHistoryRegistry const& processHistoryRegistry,
                            EventSelectionIDVector&& eventSelectionIDs,
                            BranchListIndexes&& branchListIndexes,
                            boost::shared_ptr<ProductProvenanceRetriever> mapper = boost::shared_ptr<ProductProvenanceRetriever>(new ProductProvenanceRetriever),
                            DelayedReader* reader = 0);

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

    void setLuminosityBlockPrincipal(boost::shared_ptr<LuminosityBlockPrincipal> const& lbp);

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

    boost::shared_ptr<ProductProvenanceRetriever> branchMapperPtr() const {return branchMapperPtr_;}

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler() const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    BranchListIndexes const& branchListIndexes() const;

    Provenance
    getProvenance(ProductID const& pid, ModuleCallingContext const* mcc) const;

    BasicHandle
    getByProductID(ProductID const& oid) const;

    void put(
        BranchDescription const& bd,
        WrapperOwningHolder const& edp,
        ProductProvenance const& productProvenance);

    void putOnRead(
        BranchDescription const& bd,
        void const* product,
        ProductProvenance const& productProvenance);

    WrapperHolder getIt(ProductID const& pid) const;

    ProductID branchIDToProductID(BranchID const& bid) const;

    void mergeMappers(EventPrincipal const& other) {
      branchMapperPtr_->mergeMappers(other.branchMapperPtr());
    }

    using Base::getProvenance;

  private:

    BranchID pidToBid(ProductID const& pid) const;

    virtual bool unscheduledFill(std::string const& moduleLabel,
                                 ModuleCallingContext const* mcc) const override;

    virtual void resolveProduct_(ProductHolderBase const& phb,
                                 bool fillOnDemand,
                                 ModuleCallingContext const* mcc) const override;

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

    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;

    // Pointer to the 'mapper' that will get provenance information from the persistent store.
    boost::shared_ptr<ProductProvenanceRetriever> branchMapperPtr_;

    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;

    EventSelectionIDVector eventSelectionIDs_;

    boost::shared_ptr<BranchIDListHelper const> branchIDListHelper_;

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

