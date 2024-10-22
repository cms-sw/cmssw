#ifndef FWCore_Common_SubProcessBlockHelper_h
#define FWCore_Common_SubProcessBlockHelper_h

/** \class edm::SubProcessBlockHelper

\author W. David Dagenhart, created 4 January, 2021

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Common/interface/ProcessBlockHelperBase.h"

#include <vector>

namespace edm {

  class SubProcessBlockHelper : public ProcessBlockHelperBase {
  public:
    ProcessBlockHelperBase const* topProcessBlockHelper() const final;
    std::vector<std::string> const& topProcessesWithProcessBlockProducts() const final;
    unsigned int nProcessesInFirstFile() const final;
    std::vector<std::vector<unsigned int>> const& processBlockCacheIndices() const final;
    std::vector<std::vector<unsigned int>> const& nEntries() const final;
    std::vector<unsigned int> const& cacheIndexVectorsPerFile() const final;
    std::vector<unsigned int> const& cacheEntriesPerFile() const final;
    unsigned int processBlockIndex(std::string const& processName, EventToProcessBlockIndexes const&) const final;
    unsigned int outerOffset() const final;

    void updateFromParentProcess(ProcessBlockHelperBase const& parentProcessBlockHelper, ProductRegistry const&);

  private:
    ProcessBlockHelperBase const* topProcessBlockHelper_ = nullptr;
  };
}  // namespace edm
#endif
