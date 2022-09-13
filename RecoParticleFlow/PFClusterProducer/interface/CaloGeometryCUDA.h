#ifndef RecoParticleFlow_PFClusterProducer_interface_CaloGeometryCUDA_h
#define RecoParticleFlow_PFClusterProducer_interface_CaloGeometryCUDA_h

#include <cstdint>
#include <optional>
#include <ostream>

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

  }  // namespace geometry

}  // namespace calo

#endif  // RecoParticleFlow_PFClusterProducer_interface_CaloGeometryCUDA_h
