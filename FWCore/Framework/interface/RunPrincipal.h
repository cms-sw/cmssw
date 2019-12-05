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

#include <memory>

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Framework/interface/Principal.h"

namespace edm {

  class HistoryAppender;
  class MergeableRunProductProcesses;
  class MergeableRunProductMetadata;
  class ModuleCallingContext;
  class ProcessHistoryRegistry;

  class RunPrincipal : public Principal {
  public:
    typedef RunAuxiliary Auxiliary;
    typedef Principal Base;

    RunPrincipal(std::shared_ptr<RunAuxiliary> aux,
                 std::shared_ptr<ProductRegistry const> reg,
                 ProcessConfiguration const& pc,
                 HistoryAppender* historyAppender,
                 unsigned int iRunIndex,
                 bool isForPrimaryProcess = true,
                 MergeableRunProductProcesses const* mergeableRunProductProcesses = nullptr);
    ~RunPrincipal() override;

    void fillRunPrincipal(ProcessHistoryRegistry const& processHistoryRegistry, DelayedReader* reader = nullptr);

    /** Multiple Runs may be processed simultaneously. The
     return value can be used to identify a particular Run.
     The value will range from 0 to one less than
     the maximum number of allowed simultaneous Runs. A particular
     value will be reused once the processing of the previous Run 
     using that index has been completed.
     */
    RunIndex index() const { return index_; }

    RunAuxiliary const& aux() const { return *aux_; }

    RunNumber_t run() const { return aux().run(); }

    ProcessHistoryID const& reducedProcessHistoryID() const { return m_reducedHistoryID; }

    RunID const& id() const { return aux().id(); }

    Timestamp const& beginTime() const { return aux().beginTime(); }

    Timestamp const& endTime() const { return aux().endTime(); }

    void setEndTime(Timestamp const& time) { aux_->setEndTime(time); }

    void mergeAuxiliary(RunAuxiliary const& aux) { return aux_->mergeAuxiliary(aux); }

    void put(BranchDescription const& bd, std::unique_ptr<WrapperBase> edp) const;

    void put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const;

    MergeableRunProductMetadata* mergeableRunProductMetadata() { return mergeableRunProductMetadataPtr_.get(); }

    void preReadFile();

  private:
    unsigned int transitionIndex_() const override;

    edm::propagate_const<std::shared_ptr<RunAuxiliary>> aux_;
    ProcessHistoryID m_reducedHistoryID;
    RunIndex index_;

    // For the primary input RunPrincipals created by the EventProcessor,
    // there should be one MergeableRunProductMetadata object created
    // per concurrent run. In all other cases, this should just be null.
    edm::propagate_const<std::unique_ptr<MergeableRunProductMetadata>> mergeableRunProductMetadataPtr_;
  };
}  // namespace edm
#endif
