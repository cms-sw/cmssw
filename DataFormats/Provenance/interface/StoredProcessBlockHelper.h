#ifndef DataFormats_Provenance_StoredProcessBlockHelper_h
#define DataFormats_Provenance_StoredProcessBlockHelper_h

/** \class edm::StoredProcessBlockHelper

This contains the information from the ProcessBlockHelper
that is persistent. The processBlockCacheIndices_ data
member is a flattened  version of the vector of vectors
in the ProcessBlockHelper. It is flattened mainly for
I/O performance reasons. This is intended to be directly
used only by the IOPool code responsible for reading
and writing the persistent files. Everything else should
interact with the ProcessBlockHelper.

\author W. David Dagenhart, created 1 Oct, 2020

*/

#include <string>
#include <utility>
#include <vector>

namespace edm {

  class StoredProcessBlockHelper {
  public:
    // This constructor exists for ROOT I/O
    StoredProcessBlockHelper();

    explicit StoredProcessBlockHelper(std::vector<std::string> const& processesWithProcessBlockProducts);

    std::vector<std::string> const& processesWithProcessBlockProducts() const {
      return processesWithProcessBlockProducts_;
    }
    void setProcessesWithProcessBlockProducts(std::vector<std::string> val) {
      processesWithProcessBlockProducts_ = std::move(val);
    }

    std::vector<unsigned int> const& processBlockCacheIndices() const { return processBlockCacheIndices_; }
    void setProcessBlockCacheIndices(std::vector<unsigned int> val) { processBlockCacheIndices_ = std::move(val); }

  private:
    std::vector<std::string> processesWithProcessBlockProducts_;

    std::vector<unsigned int> processBlockCacheIndices_;
  };
}  // namespace edm
#endif
