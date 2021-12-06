#ifndef FWCore_Common_ProcessBlockHelperBase_h
#define FWCore_Common_ProcessBlockHelperBase_h

/** \class edm::ProcessBlockHelperBase

\author W. David Dagenhart, created 30 December, 2020

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Utilities/interface/FWCoreUtiliesFwd.h"

#include <string>
#include <vector>

namespace edm {

  class ProcessBlockHelperBase {
  public:
    virtual ~ProcessBlockHelperBase();

    std::vector<std::string> const& processesWithProcessBlockProducts() const {
      return processesWithProcessBlockProducts_;
    }
    void setProcessesWithProcessBlockProducts(std::vector<std::string> const& val) {
      processesWithProcessBlockProducts_ = val;
    }
    void emplaceBackProcessName(std::string const& processName) {
      processesWithProcessBlockProducts_.emplace_back(processName);
    }

    std::vector<std::string> const& addedProcesses() const { return addedProcesses_; }
    void setAddedProcesses(std::vector<std::string> const& val) { addedProcesses_ = val; }
    void emplaceBackAddedProcessName(std::string const& processName) { addedProcesses_.emplace_back(processName); }

    void updateForNewProcess(ProductRegistry const&, std::string const& processName);

    // In the function names below, top implies associated the helper associated
    // with the EventProcessor, not a helper associated with a SubProcess.

    virtual ProcessBlockHelperBase const* topProcessBlockHelper() const = 0;
    virtual std::vector<std::string> const& topProcessesWithProcessBlockProducts() const = 0;
    virtual unsigned int nProcessesInFirstFile() const = 0;
    virtual std::vector<std::vector<unsigned int>> const& processBlockCacheIndices() const = 0;
    virtual std::vector<std::vector<unsigned int>> const& nEntries() const = 0;
    virtual std::vector<unsigned int> const& cacheIndexVectorsPerFile() const = 0;
    virtual std::vector<unsigned int> const& cacheEntriesPerFile() const = 0;
    virtual unsigned int processBlockIndex(std::string const& processName, EventToProcessBlockIndexes const&) const = 0;
    virtual unsigned int outerOffset() const = 0;

    std::string selectProcess(ProductRegistry const&, ProductLabels const&, TypeID const&) const;

    static constexpr unsigned int invalidCacheIndex() { return 0xffffffff; }
    static constexpr unsigned int invalidProcessIndex() { return 0xffffffff; }

  private:
    // Includes processes with ProcessBlock branches present
    // in the first input file and not dropped on input. At
    // each processing step the new process will be added at
    // the end if there are non-transient ProcessBlock products
    // being produced. Output modules will write a copy of this
    // to persistent storage after removing any process without
    // at least one kept ProcessBlock branch.
    std::vector<std::string> processesWithProcessBlockProducts_;

    // This will have 0 or 1 element depending whether there are any
    // non-transient ProcessBlock products produced in the current
    // process (except for SubProcesses where this might have more
    // than 1 element)
    std::vector<std::string> addedProcesses_;
  };
}  // namespace edm
#endif
