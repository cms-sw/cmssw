
#include "L1Trigger/DemonstratorTools/interface/LinkId.h"

namespace l1t::demo {

  bool operator<(const LinkId& x, const LinkId& y) {
    int c = x.interface.compare(y.interface);
    return c == 0 ? (x.channel < y.channel) : (c < 0);
  }

}  // namespace l1t::demo