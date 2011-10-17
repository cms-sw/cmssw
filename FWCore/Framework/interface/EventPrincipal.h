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
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "FWCore/Framework/interface/Principal.h"

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {
  class BranchMapper;
  class DelayedReader;
  class EventID;
  class HistoryAppender;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
  public:
    typedef EventAuxiliary Auxiliary;
    typedef Principal Base;

    typedef Base::ConstGroupPtr ConstGroupPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(
        boost::shared_ptr<ProductRegistry const> reg,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender = 0);
    ~EventPrincipal() {}

    void fillEventPrincipal(EventAuxiliary const& aux,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
        boost::shared_ptr<EventSelectionIDVector> eventSelectionIDs = boost::shared_ptr<EventSelectionIDVector>(),
        boost::shared_ptr<BranchListIndexes> branchListIndexes = boost::shared_ptr<BranchListIndexes>(),
        boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
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

    LuminosityBlockNumber_t luminosityBlock() const {
      return id().luminosityBlock();
    }

    RunNumber_t run() const {
      return id().run();
    }

    RunPrincipal const& runPrincipal() const;

    RunPrincipal & runPrincipal();

    boost::shared_ptr<BranchMapper> branchMapperPtr() const {return branchMapperPtr_;}

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler() const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    BranchListIndexes const& branchListIndexes() const;

    Provenance
    getProvenance(ProductID const& pid) const;

    BasicHandle
    getByProductID(ProductID const& oid) const;

    void put(
        ConstBranchDescription const& bd,
        WrapperOwningHolder const& edp,
        ProductProvenance const& productProvenance);

    void putOnRead(
        ConstBranchDescription const& bd,
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

    virtual bool unscheduledFill(std::string const& moduleLabel) const;

    virtual void resolveProduct_(Group const& g, bool fillOnDemand) const;

  private:

    EventAuxiliary aux_;

    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;

    // Pointer to the 'mapper' that will get provenance information from the persistent store.
    boost::shared_ptr<BranchMapper> branchMapperPtr_;

    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;

    boost::shared_ptr<EventSelectionIDVector> eventSelectionIDs_;

    boost::shared_ptr<BranchListIndexes> branchListIndexes_;

    std::map<BranchListIndex, ProcessIndex> branchListIndexToProcessIndex_;

  };

  inline
  bool
  isSameEvent(EventPrincipal const& a, EventPrincipal const& b) {
    return isSameEvent(a.aux(), b.aux());
  }
}
#endif

