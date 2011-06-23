#ifndef FWCore_Framework_RunPrincipal_h
#define FWCore_Framework_RunPrincipal_h

/*----------------------------------------------------------------------

RunPrincipal: This is the class responsible for management of
per run EDProducts. It is not seen by reconstruction code;
such code sees the Run class, which is a proxy for RunPrincipal.

The major internal component of the RunPrincipal
is the DataBlock.

----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {
  class UnscheduledHandler;
  class RunPrincipal : public Principal {
  public:
    typedef RunAuxiliary Auxiliary;
    typedef Principal Base;

    RunPrincipal(
        boost::shared_ptr<RunAuxiliary> aux,
        boost::shared_ptr<ProductRegistry const> reg,
        ProcessConfiguration const& pc);
    ~RunPrincipal() {}

    void fillRunPrincipal(
        boost::shared_ptr<BranchMapper> mapper = boost::shared_ptr<BranchMapper>(),
        DelayedReader* reader = 0);

    RunAuxiliary const& aux() const {
      return *aux_;
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
      aux_->setEndTime(time);
    }

    void mergeAuxiliary(RunAuxiliary const& aux) {
      return aux_->mergeAuxiliary(aux);
    }

    // ----- Mark this RunPrincipal as having been updated in the current Process.
    void addToProcessHistory();

    void checkProcessHistory() const;

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void put(
        ConstBranchDescription const& bd,
        WrapperHolder const& edp,
        ProductProvenance& productProvenance);

    void readImmediate() const;

  private:

    virtual bool unscheduledFill(std::string const&) const {return false;}

    void resolveProductImmediate(Group const& g) const;

    // A vector of groups.
    boost::shared_ptr<RunAuxiliary> aux_;
  };
}
#endif

