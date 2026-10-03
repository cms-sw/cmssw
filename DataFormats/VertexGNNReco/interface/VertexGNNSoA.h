#ifndef DataFormats_VertexGNNReco_interface_VertexGNNSoA_h
#define DataFormats_VertexGNNReco_interface_VertexGNNSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace vertexgnn {

  constexpr int kNumSlots = 220;
  constexpr int kNumFeatures = 13;

  using SlotVector = Eigen::Vector<float, kNumSlots>;
  using PIDVector = Eigen::Vector<float, 3>;

  GENERATE_SOA_LAYOUT(TrackFeaturesLayout,
                      SOA_COLUMN(float, vz),
                      SOA_COLUMN(float, dz),
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, mva),
                      SOA_COLUMN(float, pl),
                      SOA_COLUMN(float, t_pi),
                      SOA_COLUMN(float, t_k),
                      SOA_COLUMN(float, t_p),
                      SOA_COLUMN(float, s_pi),
                      SOA_COLUMN(float, s_k),
                      SOA_COLUMN(float, s_p),
                      SOA_COLUMN(float, has_time))

  using TrackFeaturesSoA = TrackFeaturesLayout<>;

  GENERATE_SOA_LAYOUT(GNNOutputLayout,
                      SOA_EIGEN_COLUMN(SlotVector, A),
                      SOA_EIGEN_COLUMN(SlotVector, z_hat),
                      SOA_EIGEN_COLUMN(SlotVector, t_hat),
                      SOA_EIGEN_COLUMN(SlotVector, p),
                      SOA_EIGEN_COLUMN(PIDVector, pi))

  using GNNOutputSoA = GNNOutputLayout<>;

}  // namespace vertexgnn

#endif
