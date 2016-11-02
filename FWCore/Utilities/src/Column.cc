#include "FWCore/Utilities/interface/Column.h"

#include <algorithm>

namespace edm {

  Column::Column(std::string const& t)
    : Column{t,0}
  {}

  Column::Column(std::string const& t,
                 std::size_t const w)
    : title_{t}
    , width_{std::max(w, title_.size())}
  {}

}
