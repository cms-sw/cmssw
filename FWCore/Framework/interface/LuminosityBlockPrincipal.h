#ifndef FWCore_Framework_LuminosityBlockPrincipal_h
#define FWCore_Framework_LuminosityBlockPrincipal_h

/*----------------------------------------------------------------------

LuminosityBlockPrincipal: This is the class responsible for management of
per luminosity block EDProducts. It is not seen by reconstruction code;
such code sees the LuminosityBlock class, which is a proxy for LuminosityBlockPrincipal.

The major internal component of the LuminosityBlockPrincipal
is the DataBlock.

----------------------------------------------------------------------*/


#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>

#include <vector>

namespace edm {

  class HistoryAppender;
  class ModuleCallingContext;
  class ProcessHistoryRegistry;
  class RunPrincipal;

  class LuminosityBlockPrincipal : public Principal {
  public:
    typedef LuminosityBlockAuxiliary Auxiliary;
    typedef Principal Base;
    LuminosityBlockPrincipal(
        std::shared_ptr<LuminosityBlockAuxiliary> aux,
        std::shared_ptr<ProductRegistry const> reg,
        ProcessConfiguration const& pc,
        HistoryAppender* historyAppender,
        unsigned int index,
        bool isForPrimaryProcess=true);

    ~LuminosityBlockPrincipal() override {}

    void fillLuminosityBlockPrincipal(ProcessHistoryRegistry const& processHistoryRegistry, DelayedReader* reader = nullptr);

    RunPrincipal const& runPrincipal() const {
      return *runPrincipal_;
    }

    RunPrincipal& runPrincipal() {
      return *runPrincipal_;
    }

    void setRunPrincipal(std::shared_ptr<RunPrincipal> rp) {
      runPrincipal_ = rp;
    }

    LuminosityBlockIndex index() const {
      return index_;
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

    void mergeAuxiliary(LuminosityBlockAuxiliary const& aux) {
      return aux_->mergeAuxiliary(aux);
    }

    void put(
        BranchDescription const& bd,
        std::unique_ptr<WrapperBase> edp) const;


    void setComplete() {
      complete_ = true;
    }

  private:

    bool isComplete_() const override {return complete_;}

    unsigned int transitionIndex_() const override;

    edm::propagate_const<std::shared_ptr<RunPrincipal>> runPrincipal_;

    edm::propagate_const<std::shared_ptr<LuminosityBlockAuxiliary>> aux_;

    LuminosityBlockIndex index_;
    
    bool complete_;
  };
}
#endif

