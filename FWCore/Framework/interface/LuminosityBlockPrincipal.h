#ifndef FWCore_Framework_LuminosityBlockPrincipal_h
#define FWCore_Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------
  
LuminosityBlockPrincipal: This is the class responsible for management of
per luminosity block EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

$Id: LuminosityBlockPrincipal.h,v 1.33 2008/07/03 04:32:39 wmtan Exp $

----------------------------------------------------------------------*/

#include "boost/shared_ptr.hpp"
#include <vector>

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {
  class RunPrincipal;
  class UnscheduledHandler;
  class LuminosityBlockPrincipal : public Principal {
  public:
    typedef LuminosityBlockAuxiliary Auxiliary;
    typedef std::vector<LumiEntryInfo> EntryInfoVector;
    typedef Principal Base;
    LuminosityBlockPrincipal(LuminosityBlockAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist = ProcessHistoryID(),
	boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    ~LuminosityBlockPrincipal() {}

    RunPrincipal const& runPrincipal() const {
      return *runPrincipal_;
    }

    RunPrincipal & runPrincipal() {
      return *runPrincipal_;
    }

    boost::shared_ptr<RunPrincipal>
    runPrincipalSharedPtr() {
      return runPrincipal_;
    }

    void setRunPrincipal(boost::shared_ptr<RunPrincipal> rp) {
      runPrincipal_ = rp;
    }

    LuminosityBlockID id() const {
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

    LuminosityBlockNumber_t luminosityBlock() const {
      return aux().luminosityBlock();
    }

    LuminosityBlockAuxiliary const& aux() const {
      aux_.processHistoryID_ = processHistoryID();
      return aux_;
    }

    RunNumber_t run() const {
      return aux().run();
    }

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void mergeLuminosityBlock(boost::shared_ptr<LuminosityBlockPrincipal> lbp);

    Provenance
    getProvenance(BranchID const& bid) const;

    void
    getAllProvenance(std::vector<Provenance const *> & provenances) const;

    void put(std::auto_ptr<EDProduct> edp,
	     ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd);

    void addGroup(std::auto_ptr<EDProduct> prod, ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

    void addGroup(ConstBranchDescription const& bd, std::auto_ptr<EventEntryInfo> entryInfo);

  private:
    virtual void addOrReplaceGroup(std::auto_ptr<Group> g);

    virtual void resolveProvenance(Group const& g) const;

    virtual bool unscheduledFill(std::string const&) const {return false;}

    boost::shared_ptr<RunPrincipal> runPrincipal_;
    LuminosityBlockAuxiliary aux_;
  };
}
#endif

