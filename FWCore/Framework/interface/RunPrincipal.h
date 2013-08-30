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

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {

  class HistoryAppender;
  class ModuleCallingContext;
  class UnscheduledHandler;

  class RunPrincipal : public Principal {
  public:
    typedef RunAuxiliary Auxiliary;
    typedef Principal Base;

    RunPrincipal(
        boost::shared_ptr<RunAuxiliary> aux,
        boost::shared_ptr<ProductRegistry const> reg,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender,
        unsigned int iRunIndex);
    ~RunPrincipal() {}

    void fillRunPrincipal(ProcessHistoryRegistry& processHistoryRegistry, DelayedReader* reader = 0);

    /** Multiple Runs may be processed simultaneously. The
     return value can be used to identify a particular Run.
     The value will range from 0 to one less than
     the maximum number of allowed simultaneous Runs. A particular
     value will be reused once the processing of the previous Run 
     using that index has been completed.
     */
    RunIndex index() const {
      return index_;
    }
    
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

    void setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler>) {}

    void put(
        BranchDescription const& bd,
        WrapperOwningHolder const& edp);

    void readImmediate() const;

    void setComplete() {
      complete_ = true;
    }

  private:

    virtual bool isComplete_() const override {return complete_;}

    virtual bool unscheduledFill(std::string const&,
                                 ModuleCallingContext const* mcc) const override {return false;}

    void resolveProductImmediate(ProductHolderBase const& phb) const;

    // A vector of product holders.
    boost::shared_ptr<RunAuxiliary> aux_;
    RunIndex index_;

    bool complete_;
  };
}
#endif

