#ifndef DataFormats_Provenance_StoredProcessBlockHelper_h
#define DataFormats_Provenance_StoredProcessBlockHelper_h

/** \class edm::StoredProcessBlockHelper

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

  private:
    std::vector<std::string> processesWithProcessBlockProducts_;
  };
}  // namespace edm
#endif
