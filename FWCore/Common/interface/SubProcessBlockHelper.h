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
    ProcessBlockHelperBase const* topProcessBlockHelper() const override;
    std::vector<std::string> const& topProcessesWithProcessBlockProducts() const override;
    unsigned int nProcessesInFirstFile() const override;
    std::vector<std::vector<unsigned int>> const& processBlockCacheIndices() const override;
    std::vector<std::vector<unsigned int>> const& nEntries() const override;
    std::vector<unsigned int> const& cacheIndexVectorsPerFile() const override;
    std::vector<unsigned int> const& cacheEntriesPerFile() const override;
    unsigned int processBlockIndex(std::string const& processName, EventToProcessBlockIndexes const&) const override;
    unsigned int outerOffset() const override;

    void updateFromParentProcess(ProcessBlockHelperBase const& parentProcessBlockHelper, ProductRegistry const&);

  private:
    ProcessBlockHelperBase const* topProcessBlockHelper_ = nullptr;
  };
}  // namespace edm
#endif
