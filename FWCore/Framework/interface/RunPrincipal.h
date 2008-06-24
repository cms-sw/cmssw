#ifndef FWCore_Framework_RunPrincipal_h
#define FWCore_Framework_RunPrincipal_h

/*----------------------------------------------------------------------
  
RunPrincipal: This is the class responsible for management of
per run EDProducts. It is not seen by reconstruction code;
such code sees the Run class, which is a proxy for RunPrincipal.

The major internal component of the RunPrincipal
is the DataBlock.

$Id: RunPrincipal.h,v 1.24 2008/05/12 18:14:07 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"
#include <vector>

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {
  class UnscheduledHandler;
  class RunPrincipal : public Principal<RunLumiEntryInfo> {
  public:
    typedef RunAuxiliary Auxiliary;
    typedef LumiEntryInfo EntryInfo;
    typedef BranchMapper<EntryInfo> Mapper;
    typedef std::vector<EntryInfo> EntryInfoVector;
    typedef Principal<EntryInfo> Base;

    RunPrincipal(RunAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<Mapper> mapper = boost::shared_ptr<Mapper>(new Mapper),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader)) :
	  Base(reg, pc, hist, mapper, rtrv),
	  aux_(aux) {}
    ~RunPrincipal() {}

    RunAuxiliary const& aux() const {
      aux_.processHistoryID_ = processHistoryID();
      return aux_;
    }

    RunNumber_t run() const {
      return aux().run();
    }

    RunID const& id() const {
      return aux().id();
    }

    Timestamp const& beginTime() const {
      return aux().beginTime();
    }

    Timestamp const& endTime() const {
      return aux().endTime();
    }

    void setEndTime(Timestamp const& time) {
      aux_.setEndTime(time);
    }

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void mergeRun(boost::shared_ptr<RunPrincipal> rp);

    Provenance
    getProvenance(BranchID const& bid) const;

    void
    getAllProvenance(std::vector<Provenance const *> & provenances) const;

    void put(std::auto_ptr<EDProduct> edp,
	     ConstBranchDescription const& bd, std::auto_ptr<RunLumiEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd);

    void addGroup(std::auto_ptr<EDProduct> prod, ConstBranchDescription const& bd, std::auto_ptr<RunLumiEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd, std::auto_ptr<RunLumiEntryInfo> entryInfo);

  private:

    virtual void addOrReplaceGroup(std::auto_ptr<Group> g);

    virtual void resolveProvenance(Group const& g) const;

    virtual bool unscheduledFill(std::string const&) const {return false;}

    RunAuxiliary aux_;
  };
}
#endif

