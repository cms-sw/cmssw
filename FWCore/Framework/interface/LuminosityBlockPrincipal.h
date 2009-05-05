#ifndef FWCore_Framework_LuminosityBlockPrincipal_h
#define FWCore_Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------
  
LuminosityBlockPrincipal: This is the class responsible for management of
per luminosity block EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

$Id: LuminosityBlockPrincipal.h,v 1.36 2009/04/15 23:22:30 wmtan Exp $

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
    typedef std::vector<ProductProvenance> EntryInfoVector;
    typedef Principal Base;
    LuminosityBlockPrincipal(LuminosityBlockAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
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
      return aux_;
    }

    RunNumber_t run() const {
      return aux().run();
    }

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void mergeLuminosityBlock(boost::shared_ptr<LuminosityBlockPrincipal> lbp);

    void put(boost::shared_ptr<EDProduct> edp,
	     ConstBranchDescription const& bd, std::auto_ptr<ProductProvenance> productProvenance);

    void addGroup(ConstBranchDescription const& bd);

    void addGroup(boost::shared_ptr<EDProduct> prod, ConstBranchDescription const& bd, std::auto_ptr<ProductProvenance> productProvenance);

    void addGroup(ConstBranchDescription const& bd, std::auto_ptr<ProductProvenance> productProvenance);

    void swap(LuminosityBlockPrincipal&);

  private:
    virtual void addOrReplaceGroup(std::auto_ptr<Group> g);

    virtual ProcessHistoryID const& processHistoryID() const {return aux().processHistoryID_;}

    virtual void setProcessHistoryID(ProcessHistoryID const& phid) const {return aux().setProcessHistoryID(phid);}

    virtual bool unscheduledFill(std::string const&) const {return false;}

    boost::shared_ptr<RunPrincipal> runPrincipal_;
    LuminosityBlockAuxiliary aux_;
  };
}
#endif

