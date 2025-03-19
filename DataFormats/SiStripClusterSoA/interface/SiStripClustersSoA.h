
#ifndef DataFormats_SiStripClusterSoA_interface_SiStripClustersSoA_h
#define DataFormats_SiStripClusterSoA_interface_SiStripClustersSoA_h

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sistrip {
  const static auto maxStripsPerCluster = 768;
  using clusterADCsColumn = edm::StdArray<uint8_t, maxStripsPerCluster>; /*768*/

  GENERATE_SOA_LAYOUT(SiStripClustersSoALayout,
                      SOA_COLUMN(uint32_t, clusterIndex),
                      SOA_COLUMN(uint32_t, clusterSize),
                      SOA_COLUMN(clusterADCsColumn, clusterADCs),
                      SOA_COLUMN(uint32_t, clusterDetId),
                      SOA_COLUMN(uint16_t, firstStrip),
                      SOA_COLUMN(bool, trueCluster),
                      SOA_COLUMN(float, barycenter),
                      SOA_COLUMN(float, charge),
                      //
                      SOA_SCALAR(uint32_t, nClusters),
                      SOA_SCALAR(uint32_t, maxClusterSize))

  using SiStripClustersSoA = SiStripClustersSoALayout<>;
  using SiStripClustersView = SiStripClustersSoA::View;
  using SiStripClustersConstView = SiStripClustersSoA::ConstView;
}  // namespace sistrip

#endif  // DataFormats_SiStripClusterSoA_interface_SiStripClustersSoA_h
