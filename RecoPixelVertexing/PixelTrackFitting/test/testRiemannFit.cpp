#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef USE_BL
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"
#else
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#endif

#include "test_common.h"

using namespace Eigen;

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 5 * 1024; }
  constexpr uint32_t stride() { return maxNumberOfTracks(); }
  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()> >;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()> >;
  // fast fit
  using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()> >;

}  // namespace Rfit

/*
Hit global: 641,0 2: 2.934787,0.773211,-10.980247
Error: 641,0 2: 1.424715e-07,-4.996975e-07,1.752614e-06,3.660689e-11,1.644638e-09,7.346080e-05
Hit global: 641,1 104: 6.314229,1.816356,-23.162731
Error: 641,1 104: 6.899177e-08,-1.873414e-07,5.087101e-07,-2.078806e-10,-2.210498e-11,4.346079e-06
Hit global: 641,2 1521: 8.936963,2.765734,-32.759060
Error: 641,2 1521: 1.406273e-06,4.042467e-07,6.391180e-07,-3.141497e-07,6.513821e-08,1.163863e-07
Hit global: 641,3 1712: 10.360559,3.330824,-38.061260
Error: 641,3 1712: 1.176358e-06,2.154100e-07,5.072816e-07,-8.161219e-08,1.437878e-07,5.951832e-08
Hit global: 641,4 1824: 12.856387,4.422212,-47.518867
Error: 641,4 1824: 2.852843e-05,7.956492e-06,3.117701e-06,-1.060541e-06,8.777413e-09,1.426417e-07
*/

template <typename M3xN, typename M6xN>
void fillHitsAndHitsCov(M3xN& hits, M6xN& hits_ge) {
  constexpr uint32_t N = M3xN::ColsAtCompileTime;

  if (N == 5) {
    hits << 2.934787, 6.314229, 8.936963, 10.360559, 12.856387, 0.773211, 1.816356, 2.765734, 3.330824, 4.422212,
        -10.980247, -23.162731, -32.759060, -38.061260, -47.518867;
    hits_ge.col(0) << 1.424715e-07, -4.996975e-07, 1.752614e-06, 3.660689e-11, 1.644638e-09, 7.346080e-05;
    hits_ge.col(1) << 6.899177e-08, -1.873414e-07, 5.087101e-07, -2.078806e-10, -2.210498e-11, 4.346079e-06;
    hits_ge.col(2) << 1.406273e-06, 4.042467e-07, 6.391180e-07, -3.141497e-07, 6.513821e-08, 1.163863e-07;
    hits_ge.col(3) << 1.176358e-06, 2.154100e-07, 5.072816e-07, -8.161219e-08, 1.437878e-07, 5.951832e-08;
    hits_ge.col(4) << 2.852843e-05, 7.956492e-06, 3.117701e-06, -1.060541e-06, 8.777413e-09, 1.426417e-07;
    return;
  }

  if (N > 3)
    hits << 1.98645, 4.72598, 7.65632, 11.3151, 2.18002, 4.88864, 7.75845, 11.3134, 2.46338, 6.99838, 11.808, 17.793;
  else
    hits << 1.98645, 4.72598, 7.65632, 2.18002, 4.88864, 7.75845, 2.46338, 6.99838, 11.808;

  hits_ge.col(0)[0] = 7.14652e-06;
  hits_ge.col(1)[0] = 2.15789e-06;
  hits_ge.col(2)[0] = 1.63328e-06;
  if (N > 3)
    hits_ge.col(3)[0] = 6.27919e-06;
  hits_ge.col(0)[2] = 6.10348e-06;
  hits_ge.col(1)[2] = 2.08211e-06;
  hits_ge.col(2)[2] = 1.61672e-06;
  if (N > 3)
    hits_ge.col(3)[2] = 6.28081e-06;
  hits_ge.col(0)[5] = 5.184e-05;
  hits_ge.col(1)[5] = 1.444e-05;
  hits_ge.col(2)[5] = 6.25e-06;
  if (N > 3)
    hits_ge.col(3)[5] = 3.136e-05;
  hits_ge.col(0)[1] = -5.60077e-06;
  hits_ge.col(1)[1] = -1.11936e-06;
  hits_ge.col(2)[1] = -6.24945e-07;
  if (N > 3)
    hits_ge.col(3)[1] = -5.28e-06;
}

template <int N>
void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd<N> hits;
  Rfit::Matrix6xNf<N> hits_ge = MatrixXf::Zero(6, N);

  fillHitsAndHitsCov(hits, hits_ge);

  std::cout << "sizes " << N << ' ' << sizeof(hits) << ' ' << sizeof(hits_ge) << ' ' << sizeof(Vector4d) << std::endl;

  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

  // FAST_FIT_CPU
#ifdef USE_BL
  Vector4d fast_fit_results;
  BrokenLine::BL_Fast_fit(hits, fast_fit_results);
#else
  Vector4d fast_fit_results;
  Rfit::Fast_fit(hits, fast_fit_results);
#endif
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;

  // CIRCLE_FIT CPU

#ifdef USE_BL
  BrokenLine::PreparedBrokenLineData<N> data;
  BrokenLine::karimaki_circle_fit circle_fit_results;
  Rfit::Matrix3d Jacob;

  BrokenLine::prepareBrokenLineData(hits, fast_fit_results, B, data);
  Rfit::line_fit line_fit_results;
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_results, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_results, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();
#else
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge, hits_cov);
  Rfit::circle_fit circle_fit_results =
      Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit_results, rad, B, true);
  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);
  Rfit::par_uvrtopak(circle_fit_results, B, true);

#endif

  std::cout << "Fitted values (CircleFit):\n"
            << circle_fit_results.par << "\nchi2 " << circle_fit_results.chi2 << std::endl;
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << "\nchi2 " << line_fit_results.chi2 << std::endl;

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
}

int main(int argc, char* argv[]) {
  testFit<4>();
  testFit<3>();
  testFit<5>();

  return 0;
}
