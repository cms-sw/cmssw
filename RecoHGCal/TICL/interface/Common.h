#ifndef RecoHGCal_TICL_interface_Common_h
#define RecoHGCal_TICL_interface_Common_h

#include <vector>
#include <cstdint>

namespace ticl::constants {
constexpr float minEta = 1.5f;
constexpr float maxEta = 3.2f;
}  // namespace ticl::constants

namespace ticl {
  typedef std::vector<std::pair<unsigned int, float> > HgcalClusterFilterMask;
}

#endif  // RecoHGCal_TICL_interface_Common_h
