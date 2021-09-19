#ifndef RecoParticleFlow_PFClusterProducerCUDA_interface_CaloGeometryCUDA_h
#define RecoParticleFlow_PFClusterProducerCUDA_interface_CaloGeometryCUDA_h

#include <ostream>
#include <optional>
#include <cstdint>

#include "DataFormats/DetId/interface/DetId.h"

namespace calo {
  namespace geometry {
    enum Detector = DteId::Detector;
    enum CaloGeometry_Detail {
      kMaxDet = 10,
      kMinDet = 3,
      kNDets = kMaxDet - kMinDet + 1,
      kMaxSub = 6,
      kNSubDets = kMaxSub + 1,
      kLength = kNDets * kNSubDets
    };
    


  } // namespace Geometry_LUT


} // namespace calo

#endif
