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
#include <vector>

namespace edm {

  class StoredProcessBlockHelper {
  public:
    // This constructor exists for ROOT I/O
    StoredProcessBlockHelper();

    StoredProcessBlockHelper(std::vector<std::string> const& processesWithProcessBlockProducts);

    std::vector<std::string> const& processesWithProcessBlockProducts() const {
      return processesWithProcessBlockProducts_;
    }

    std::vector<std::string>& processesWithProcessBlockProducts() { return processesWithProcessBlockProducts_; }

    std::vector<unsigned int> const& processBlockCacheIndices() const { return processBlockCacheIndices_; }

    std::vector<unsigned int>& processBlockCacheIndices() { return processBlockCacheIndices_; }

  private:
    std::vector<std::string> processesWithProcessBlockProducts_;

    std::vector<unsigned int> processBlockCacheIndices_;
  };
}  // namespace edm
#endif
