#include <Eigen/Dense>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TrajectoryStateSoA_t.h"

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

using namespace cms::alpakatools;

namespace ALPAKA_ACCELERATOR_NAMESPACE::test {

  namespace {

    ALPAKA_FN_ACC Matrix5d buildCovariance(Vector5d const& e) {
      Matrix5d cov;
      for (int i = 0; i < 5; ++i)
        cov(i, i) = e(i) * e(i);
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < i; ++j) {
          // this makes the matrix positive defined
          double v = 0.3 * std::sqrt(cov(i, i) * cov(j, j));
          cov(i, j) = (i + j) % 2 ? -0.4 * v : 0.1 * v;
          cov(j, i) = cov(i, j);
        }
      }
      return cov;
    }

    template <typename TrackerTraits>
    struct TestTrackSoA {

      ALPAKA_FN_ACC void operator()(Acc1D const& acc, reco::TrackSoAView tracks) const {
        Vector5d par0;
        par0 << 0.2, 0.1, 3.5, 0.8, 0.1;
        Vector5d e0;
        e0 << 0.01, 0.01, 0.035, -0.03, -0.01;
        Matrix5d cov0 = buildCovariance(e0);

        for (auto i : uniform_elements(acc, tracks.metadata().size())) {
          reco::copyFromDense(tracks, par0, cov0, i);
          Vector5d par1;
          Matrix5d cov1;
          reco::copyToDense(tracks, par1, cov1, i);
          Vector5d deltaV = par1 - par0;
          Matrix5d deltaM = cov1 - cov0;
          for (int j = 0; j < 5; ++j) {
            ALPAKA_ASSERT(std::abs(deltaV(j)) < 1.e-5);
            for (int k = j; k < 5; ++k) {
              ALPAKA_ASSERT(cov0(k, j) == cov0(j, k));
              ALPAKA_ASSERT(cov1(k, j) == cov1(j, k));
              ALPAKA_ASSERT(std::abs(deltaM(k, j)) < 1.e-5);
            }
          }
        }
      }
    };

  }  // namespace

  template <typename TrackerTraits>
  void testTrackSoA(Queue& queue, ::reco::TrackSoAView& tracks) {
    auto grid = make_workdiv<Acc1D>(1, 64);
    alpaka::exec<Acc1D>(queue, grid, TestTrackSoA<TrackerTraits>{}, tracks);
  }

  template void testTrackSoA<pixelTopology::Phase1>(Queue& queue, reco::TrackSoAView& tracks);
  template void testTrackSoA<pixelTopology::Phase2>(Queue& queue, reco::TrackSoAView& tracks);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::test
