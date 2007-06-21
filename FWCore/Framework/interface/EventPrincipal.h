#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

$Id: EventPrincipal.h,v 1.56 2007/06/14 17:52:15 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {
  class EventID;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : private Principal {
    typedef Principal Base;
  public:
    typedef Base::SharedConstGroupPtr SharedConstGroupPtr;
    EventPrincipal(EventID const& id,
	Timestamp const& time,
	boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
        ProcessConfiguration const& pc,
        bool isReal,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));
    EventPrincipal(EventID const& id,
	Timestamp const& time,
	boost::shared_ptr<ProductRegistry const> reg,
	LuminosityBlockNumber_t lumi,
        ProcessConfiguration const& pc,
        bool isReal,
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

    EventAuxiliary const& aux() const {
      aux_.processHistoryID_ = processHistoryID();
      return aux_;
    }

    LuminosityBlockNumber_t const& luminosityBlock() const {
      return aux().luminosityBlock();
    }

    RunNumber_t runNumber() const {
      return id().run();
    }

    RunPrincipal const& runPrincipal() const;

    RunPrincipal & runPrincipal();

    using Base::addGroup;
    using Base::addToProcessHistory;
    using Base::getAllProvenance;
    using Base::getByLabel;
    using Base::get;
    using Base::getBySelector;
    using Base::getByType;
    using Base::getForOutput;
    using Base::getIt;
    using Base::getMany;
    using Base::getManyByType;
    using Base::getProvenance;
    using Base::groupGetter;
    using Base::numEDProducts;
    using Base::processHistory;
    using Base::processHistoryID;
    using Base::prodGetter;
    using Base::productRegistry;
    using Base::put;
    using Base::size;
    using Base::store;

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);

  private:

    virtual bool unscheduledFill(Provenance const& prov) const;

    EventAuxiliary aux_;
    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;
    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;
  };

  inline
  bool
  isSameRun(EventPrincipal const* a, EventPrincipal const* b) {
    return(a != 0 && b != 0 && a->runNumber() == b->runNumber());
  }

  inline
  bool
  isSameLumi(EventPrincipal const* a, EventPrincipal const* b) {
    return(isSameRun(a, b) && a->luminosityBlock() == b->luminosityBlock());
  }
}
#endif

