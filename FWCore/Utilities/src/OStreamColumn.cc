#include "FWCore/Utilities/interface/OStreamColumn.h"

#include <algorithm>

namespace edm {

  OStreamColumn::OStreamColumn(std::string const& t) : OStreamColumn{t, 0} {}

  OStreamColumn::OStreamColumn(std::string const& t, int const w)
      : title_{t}, width_{std::max(w, static_cast<int>(title_.size()))} {}

  std::ostream& operator<<(std::ostream& t, OStreamColumn const& c) {
    t << std::setw(c.width_) << c.title_;
    return t;
  }

}  // namespace edm
