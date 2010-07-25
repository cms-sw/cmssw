#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/History.h"
#include "FWCore/Framework/interface/NoDelayedReader.h"
#include "FWCore/Framework/interface/Principal.h"


namespace edm {
  class EventID;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
  public:
    typedef EventAuxiliary Auxiliary;
    typedef std::vector<ProductProvenance> EntryInfoVector;

    typedef Principal Base;

    typedef Base::SharedConstGroupPtr SharedConstGroupPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc);
    ~EventPrincipal() {}

    void fillEventPrincipal(std::auto_ptr<EventAuxiliary> aux,
	boost::shared_ptr<LuminosityBlockPrincipal> lbp,
	boost::shared_ptr<History> history = boost::shared_ptr<History>(new History),
	boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    void clearEventPrincipal();

    LuminosityBlockPrincipal const& luminosityBlockPrincipal() const {
      return *luminosityBlockPrincipal_;
    }

    LuminosityBlockPrincipal& luminosityBlockPrincipal() {
      return *luminosityBlockPrincipal_;
    }

    EventID const& id() const {
      return aux().id();
    }

    Timestamp const& time() const {
      return aux().time();
    }

    bool const isReal() const {
      return aux().isRealData();
    }

    EventAuxiliary::ExperimentType ExperimentType() const {
      return aux().experimentType();
    }

    int const bunchCrossing() const {
      return aux().bunchCrossing();
    }

    int const storeNumber() const {
      return aux().storeNumber();
    }

    EventAuxiliary const& aux() const {
      return *aux_;
    }

    LuminosityBlockNumber_t luminosityBlock() const {
      return id().luminosityBlock();
    }

    RunNumber_t run() const {
      return id().run();
    }

    RunPrincipal const& runPrincipal() const;

    RunPrincipal & runPrincipal();

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler() const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    History const& history() const {return *history_;}

    History& history() {return *history_;}

    Provenance
    getProvenance(ProductID const& pid) const;

    BasicHandle
    getByProductID(ProductID const& oid) const;

    void put(
	ConstBranchDescription const& bd,
	std::auto_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance);

    void putOnRead(
	ConstBranchDescription const& bd,
	std::auto_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance);

    virtual EDProduct const* getIt(ProductID const& pid) const;

    ProductID branchIDToProductID(BranchID const& bid) const;

    using Base::getProvenance;

  private:

    BranchID pidToBid(ProductID const& pid) const;

    virtual ProductID oldToNewProductID_(ProductID const& oldProductID) const;

    virtual bool unscheduledFill(std::string const& moduleLabel) const;

    virtual void resolveProduct_(Group const& g, bool fillOnDemand) const;

  private:

    boost::scoped_ptr<EventAuxiliary> aux_;

    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;

    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;

    boost::shared_ptr<History> history_;

    std::map<BranchListIndex, ProcessIndex> branchListIndexToProcessIndex_;

  };

  inline
  bool
  isSameEvent(EventPrincipal const& a, EventPrincipal const& b) {
    return isSameEvent(a.aux(), b.aux());
  }
}
#endif

