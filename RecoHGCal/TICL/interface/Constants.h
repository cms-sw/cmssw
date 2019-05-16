#ifndef RecoHGCal_TICL_interface_Constants_h
#define RecoHGCal_TICL_interface_Constants_h

#include <vector>
#include <cstdint>

namespace ticl::constants {
constexpr float minEta = 1.5f;
constexpr float maxEta = 3.2f;
}  // namespace ticl::constants

namespace ticl {
  typedef std::vector<std::pair<unsigned int, float> > hgcalClusterFilterMask;
}

#endif  // RecoHGCal_TICL_interface_Constants_h
