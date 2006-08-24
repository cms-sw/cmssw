#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include <ostream>

namespace edm {
  std::ostream&
  operator<<(std::ostream& os, ParameterSetBlob const& blob) {
    os << blob.pset_;
    return os;
  }
}
