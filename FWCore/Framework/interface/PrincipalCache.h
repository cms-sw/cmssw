#ifndef FWCore_Framework_PrincipalCache_h
#define FWCore_Framework_PrincipalCache_h

/*
Contains a shared pointer to the RunPrincipal,
LuminosityBlockPrincipal, and EventPrincipal.
Manages merging of run and luminosity block
principals when there is more than one principal
from the same run or luminosity block and having
the same reduced ProcessHistoryID.

The EventPrincipal is reused each event and is created
by the EventProcessor or SubProcess which contains
an object of this type as a data member.

The RunPrincipal and LuminosityBlockPrincipal is
created by the InputSource each time a different
run or luminosity block is encountered.

Performs checks that process history IDs or runs and
lumis, run numbers, and luminosity numbers are consistent.

Original Author: W. David Dagenhart
*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include <memory>
#include <vector>
#include <cassert>

namespace edm {

  class ProcessBlockPrincipal;
  class RunPrincipal;
  class LuminosityBlockPrincipal;
  class EventPrincipal;
  class RunAuxiliary;
  class LuminosityBlockAuxiliary;
  class ProductRegistry;
  class PreallocationConfiguration;

  class PrincipalCache {
  public:
    PrincipalCache();
    ~PrincipalCache();
    PrincipalCache(PrincipalCache&&) = default;

    ProcessBlockPrincipal& processBlockPrincipal() const { return *processBlockPrincipal_; }
    ProcessBlockPrincipal& inputProcessBlockPrincipal() const { return *inputProcessBlockPrincipal_; }

    enum class ProcessBlockType { New, Input };
    ProcessBlockPrincipal& processBlockPrincipal(ProcessBlockType processBlockType) const {
      return processBlockType == ProcessBlockType::Input ? *inputProcessBlockPrincipal_ : *processBlockPrincipal_;
    }

    RunPrincipal& runPrincipal(ProcessHistoryID const& phid, RunNumber_t run) const;
    std::shared_ptr<RunPrincipal> const& runPrincipalPtr(ProcessHistoryID const& phid, RunNumber_t run) const;
    RunPrincipal& runPrincipal() const;
    std::shared_ptr<RunPrincipal> const& runPrincipalPtr() const;
    bool hasRunPrincipal() const { return bool(runPrincipal_); }

    std::shared_ptr<LuminosityBlockPrincipal> getAvailableLumiPrincipalPtr();

    EventPrincipal& eventPrincipal(unsigned int iStreamIndex) const { return *(eventPrincipals_[iStreamIndex]); }

    void merge(std::shared_ptr<RunAuxiliary> aux, std::shared_ptr<ProductRegistry const> reg);

    void setNumberOfConcurrentPrincipals(PreallocationConfiguration const&);
    void insert(std::unique_ptr<ProcessBlockPrincipal>);
    void insertForInput(std::unique_ptr<ProcessBlockPrincipal>);
    void insert(std::shared_ptr<RunPrincipal> rp);
    void insert(std::unique_ptr<LuminosityBlockPrincipal> lbp);
    void insert(std::shared_ptr<EventPrincipal> ep);

    void deleteRun(ProcessHistoryID const& phid, RunNumber_t run);

    void adjustEventsToNewProductRegistry(std::shared_ptr<ProductRegistry const> reg);

    void adjustIndexesAfterProductRegistryAddition();

    void setProcessHistoryRegistry(ProcessHistoryRegistry const& phr) { processHistoryRegistry_ = &phr; }

    void preReadFile();

  private:
    void throwRunMissing() const;
    void throwLumiMissing() const;

    // These are explicitly cleared when finished with the processblock, run,
    // lumi, or event
    std::unique_ptr<ProcessBlockPrincipal> processBlockPrincipal_;
    std::unique_ptr<ProcessBlockPrincipal> inputProcessBlockPrincipal_;
    std::shared_ptr<RunPrincipal> runPrincipal_;
    edm::ReusableObjectHolder<LuminosityBlockPrincipal> lumiHolder_;
    std::vector<std::shared_ptr<EventPrincipal>> eventPrincipals_;

    // This is just an accessor to the registry owned by the input source.
    ProcessHistoryRegistry const* processHistoryRegistry_;  // We don't own this

    // These are intentionally not cleared so that when inserting
    // the next principal the conversion from full ProcessHistoryID_
    // to reduced ProcessHistoryID_ is still in memory and does
    // not need to be recalculated if the ID does not change. I
    // expect that very often these ID's will not change from one
    // principal to the next and a good amount of CPU can be saved
    // by not recalculating.
    ProcessHistoryID inputProcessHistoryID_;
    ProcessHistoryID reducedInputProcessHistoryID_;
    RunNumber_t run_;
    LuminosityBlockNumber_t lumi_;
  };
}  // namespace edm

#endif
