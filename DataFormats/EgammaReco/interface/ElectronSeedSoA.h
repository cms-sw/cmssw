#ifndef DataFormats_EgammaReco_interface_ElectronSeedSoA_h
#define DataFormats_EgammaReco_interface_ElectronSeedSoA_h

#include <Eigen/Core>
#include <cstdint>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Vector3f = Eigen::Matrix<float, 3, 1>;

  GENERATE_SOA_LAYOUT(ElectronSeedLayout,
                      SOA_COLUMN(int8_t, nHits),
                      SOA_COLUMN(int8_t, isMatched),
                      SOA_COLUMN(int16_t, matchedScID),
                      SOA_COLUMN(int32_t, id),
                      SOA_COLUMN(int16_t, hit0detectorID),
                      SOA_COLUMN(int16_t, hit0isValid),
                      SOA_COLUMN(int16_t, hit1detectorID),
                      SOA_COLUMN(int16_t, hit1isValid),
                      SOA_COLUMN(int16_t, hit2detectorID),
                      SOA_COLUMN(int16_t, hit2isValid),
                      SOA_EIGEN_COLUMN(Vector3d, hit0Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf0Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf0Rot),
                      SOA_EIGEN_COLUMN(Vector3d, hit1Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf1Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf1Rot),
                      SOA_EIGEN_COLUMN(Vector3d, hit2Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf2Pos),
                      SOA_EIGEN_COLUMN(Vector3d, surf2Rot),
                      SOA_EIGEN_COLUMN(Vector3f, PMVars_dRZPos),
                      SOA_EIGEN_COLUMN(Vector3f, PMVars_dRZNeg),
                      SOA_EIGEN_COLUMN(Vector3f, PMVars_dPhiPos),
                      SOA_EIGEN_COLUMN(Vector3f, PMVars_dPhiNeg))
  using ElectronSeedSoA = ElectronSeedLayout<>;
}  // namespace reco

#endif
