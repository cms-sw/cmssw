#include <vector>
#include "FWCore/Common/interface/EventBase.h"
#include "RecoHGCal/TICL/interface/Trackster.h"

namespace RecoHGCal_TICL {
struct dictionary {
  Trackster tr;
  std::vector<Trackster> vtr;
  edm::Wrapper<std::vector<Trackster> > edmwrtp;
};
}  // namespace RecoHGCal_TICL
