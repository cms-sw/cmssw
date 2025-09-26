
#ifndef DataFormats_SiStripClusterSoA_interface_SiStripClusterSoA_h
#define DataFormats_SiStripClusterSoA_interface_SiStripClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sistrip {

  GENERATE_SOA_LAYOUT(SiStripClusterSoALayout,
                      SOA_COLUMN(uint32_t, clusterIndex),
                      SOA_COLUMN(uint16_t, clusterSize),
                      SOA_COLUMN(uint32_t, clusterDetId),
                      SOA_COLUMN(uint16_t, firstStrip),
                      SOA_COLUMN(bool, candidateAccepted),
                      SOA_COLUMN(float, barycenter),
                      SOA_COLUMN(float, charge),
                      SOA_COLUMN(uint32_t, candidateAcceptedPrefix),
                      SOA_SCALAR(uint32_t, nClusterCandidates),
                      SOA_SCALAR(uint32_t, maxClusterSize))

  using SiStripClusterSoA = SiStripClusterSoALayout<>;
  using SiStripClusterView = SiStripClusterSoA::View;
  using SiStripClusterConstView = SiStripClusterSoA::ConstView;
}  // namespace sistrip

#endif  // DataFormats_SiStripClusterSoA_interface_SiStripClustersSoA_h
