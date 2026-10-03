#ifndef DataFormats_EgammaReco_interface_SuperClusterSoA_h
#define DataFormats_EgammaReco_interface_SuperClusterSoA_h

#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // SoA layout for supercluster
  GENERATE_SOA_LAYOUT(SuperClusterSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, scSeedTheta),
                      SOA_COLUMN(double, scPhi),
                      SOA_COLUMN(double, scR),
                      SOA_COLUMN(double, scEnergy),
                      SOA_COLUMN(int32_t, id))
  using SuperClusterSoA = SuperClusterSoALayout<>;
}  // namespace reco

#endif
