#ifndef FWCore_Framework_LuminosityBlockPrincipal_h
#define FWCore_Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------
  
LuminosityBlockPrincipal: This is the class responsible for management of
per luminosity block EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include "boost/scoped_ptr.hpp"
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
    LuminosityBlockPrincipal(
	boost::shared_ptr<LuminosityBlockAuxiliary> aux,
	boost::shared_ptr<ProductRegistry const> reg,
	ProcessConfiguration const& pc,
	boost::shared_ptr<RunPrincipal> rp);

    ~LuminosityBlockPrincipal() {}

    void fillLuminosityBlockPrincipal(
	boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(new BranchMapper),
	boost::shared_ptr<DelayedReader> rtrv = boost::shared_ptr<DelayedReader>(new NoDelayedReader));

    RunPrincipal const& runPrincipal() const {
      return *runPrincipal_;
    }

    RunPrincipal& runPrincipal() {
      return *runPrincipal_;
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
      aux_->setEndTime(time);
    }

    LuminosityBlockNumber_t luminosityBlock() const {
      return aux().luminosityBlock();
    }

    LuminosityBlockAuxiliary const& aux() const {
      return *aux_;
    }

    RunNumber_t run() const {
      return aux().run();
    }

    bool mergeAuxiliary(LuminosityBlockAuxiliary const& aux) {
      return aux_->mergeAuxiliary(aux);
    }

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void put(
	ConstBranchDescription const& bd,
	boost::shared_ptr<EDProduct> edp,
	std::auto_ptr<ProductProvenance> productProvenance);

    void readImmediate() const;

    void swap(LuminosityBlockPrincipal&);

  private:
    virtual ProcessHistoryID const& processHistoryID() const {return aux().processHistoryID_;}

    virtual void setProcessHistoryID(ProcessHistoryID const& phid) const {return aux().setProcessHistoryID(phid);}

    virtual bool unscheduledFill(std::string const&) const {return false;}

    void resolveProductImmediate(Group const& g) const;

    boost::shared_ptr<RunPrincipal> runPrincipal_;

    boost::shared_ptr<LuminosityBlockAuxiliary> aux_;
  };
}
#endif

