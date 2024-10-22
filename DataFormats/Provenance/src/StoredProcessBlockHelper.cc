#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"

namespace edm {

  StoredProcessBlockHelper::StoredProcessBlockHelper() = default;

  StoredProcessBlockHelper::StoredProcessBlockHelper(std::vector<std::string> const& processesWithProcessBlockProducts)
      : processesWithProcessBlockProducts_(processesWithProcessBlockProducts) {}

}  // namespace edm
