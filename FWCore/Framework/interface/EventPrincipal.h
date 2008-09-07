#ifndef FWCore_Framework_EventPrincipal_h
#define FWCore_Framework_EventPrincipal_h

/*----------------------------------------------------------------------
  
EventPrincipal: This is the class responsible for management of
per event EDProducts. It is not seen by reconstruction code;
such code sees the Event class, which is a proxy for EventPrincipal.

The major internal component of the EventPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"
#include <vector>

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/Framework/interface/Principal.h"


namespace edm {
  class EventID;
  class LuminosityBlockPrincipal;
  class RunPrincipal;
  class UnscheduledHandler;

  class EventPrincipal : public Principal {
  public:
    typedef EventAuxiliary Auxiliary;
    typedef std::vector<EventEntryInfo> EntryInfoVector;

    typedef Principal Base;

    typedef Base::SharedConstGroupPtr SharedConstGroupPtr;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;
    EventPrincipal(EventAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
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

    void setLuminosityBlockPrincipal(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
      luminosityBlockPrincipal_ = lbp;
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

    void addOnDemandGroup(ConstBranchDescription const& desc);

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler);

    EventSelectionIDVector const& eventSelectionIDs() const;

    History const& history() const;

    void setHistory(History const& h);

    Provenance
    getProvenance(BranchID const& bid) const;

    Provenance
    getProvenance(ProductID const& pid) const;

    void
    getAllProvenance(std::vector<Provenance const *> & provenances) const;

    BasicHandle
    getByProductID(ProductID const& oid) const;

    void put(std::auto_ptr<EDProduct> edp, ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd);

    void addGroup(std::auto_ptr<EDProduct> prod, ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

    void addGroup(std::auto_ptr<EDProduct> prod, ConstBranchDescription const& bd, boost::shared_ptr<EventEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd, boost::shared_ptr<EventEntryInfo> entryInfo);

    virtual EDProduct const* getIt(ProductID const& oid) const;

  private:

    virtual void addOrReplaceGroup(std::auto_ptr<Group> g);

    virtual void resolveProvenance(Group const& g) const;

    virtual bool unscheduledFill(std::string const& moduleLabel) const;

    EventAuxiliary aux_;
    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal_;

    // Handler for unscheduled modules
    boost::shared_ptr<UnscheduledHandler> unscheduledHandler_;

    mutable std::vector<std::string> moduleLabelsRunning_;

    History   eventHistory_;

  };

  inline
  bool
  isSameEvent(EventPrincipal const& a, EventPrincipal const& b) {
    return isSameEvent(a.aux(), b.aux());
  }
}
#endif

