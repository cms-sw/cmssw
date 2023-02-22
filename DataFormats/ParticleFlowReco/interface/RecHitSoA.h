#ifndef ParticleFlowReco_RecHitSoA_h
#define ParticleFlowReco_RecHitSoA_h

#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace portableRecHitSoA {

  GENERATE_SOA_LAYOUT(RecHitSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(double, x),
                      SOA_COLUMN(double, y),
                      SOA_COLUMN(double, z),
                      SOA_COLUMN(int32_t, id),
                      // scalars: one value for the whole structure
                      SOA_SCALAR(double, r)
  )

  using RecHitSoA = RecHitSoALayout<>;

}  // namespace portableRecHitSoA 

#endif
