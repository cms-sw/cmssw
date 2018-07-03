#include "FWCore/Utilities/interface/OStreamColumn.h"

#include <algorithm>

namespace edm {

  OStreamColumn::OStreamColumn(std::string const& t)
    : OStreamColumn{t,0}
  {}

  OStreamColumn::OStreamColumn(std::string const& t,
                               std::size_t const w)
    : title_{t}
    , width_{std::max(w, title_.size())}
  {}

  std::ostream& operator<<(std::ostream& t, OStreamColumn const& c)
  {
    t << std::setw(c.width_) << c.title_;
    return t;
  }

}
