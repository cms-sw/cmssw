#ifndef DataFormats_Provenance_SubProcessParentageHelper_h
#define DataFormats_Provenance_SubProcessParentageHelper_h

// This class is used to properly fill Parentage in SubProcesses.
// In particular it helps filling the BranchChildren container
// that is used when dropping descendants of products that
// have been dropped on input.
//
// This class is only filled for SubProcesses. Its data member
// only has entries for products produced in a prior SubProcess
// or the top level Process in the same overall process.

#include "DataFormats/Provenance/interface/BranchID.h"

#include <vector>

namespace edm {

  class ProductRegistry;

  class SubProcessParentageHelper {
  public:

    void update(SubProcessParentageHelper const& parentSubProcessParentageHelper,
                ProductRegistry const& parentProductRegistry);

    std::vector<BranchID> const& producedProducts() const {
      return producedProducts_;
    }

  private:

    std::vector<BranchID> producedProducts_;
  };
}
#endif
