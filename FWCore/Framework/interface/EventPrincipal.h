#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

$Id: EventPrincipal.h,v 1.72 2008/02/02 21:25:59 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {
  class EventID;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
    typedef Principal Base;
  public:
    typedef Base::SharedConstGroupPtr SharedConstGroupPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(EventAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
        ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));
    ~EventPrincipal() {}

    LuminosityBlockPrincipal const& luminosityBlockPrincipal() const {
      return *luminosityBlockPrincipal_;
    }

    LuminosityBlockPrincipal & luminosityBlockPrincipal() {
      return *luminosityBlockPrincipal_;
    }

    boost::shared_ptr<LuminosityBlockPrincipal>
    luminosityBlockPrincipalSharedPtr() {
      return luminosityBlockPrincipal_;
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
      aux_.processHistoryID_ = processHistoryID();
      return aux_;
    }

    LuminosityBlockNumber_t const& luminosityBlock() const {
      return aux().luminosityBlock();
    }

    RunNumber_t run() const {
      return id().run();
    }

    RunPrincipal const& runPrincipal() const;

    RunPrincipal & runPrincipal();

    void addGroup(std::auto_ptr<Provenance>);

    using Base::addGroup;

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);

  private:
    virtual void addOrReplaceGroup(std::auto_ptr<Group> g);

    virtual bool unscheduledFill(Provenance const& prov) const;

    EventAuxiliary aux_;
    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;
    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;
  };

  inline
  bool
  isSameEvent(EventPrincipal const& a, EventPrincipal const& b) {
    return isSameEvent(a.aux(), b.aux());
  }
}
#endif

