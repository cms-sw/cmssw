#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "test_common.h"

using namespace Eigen;

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 5*1024; }
  constexpr uint32_t stride() { return maxNumberOfTracks();}
  using Matrix3x4d = Eigen::Matrix<double,3,4>;
  using Map3x4d = Eigen::Map<Matrix3x4d,0,Eigen::Stride<3*stride(),stride()> >;
  using Matrix6x4f = Eigen::Matrix<float,6,4>;
  using Map6x4f = Eigen::Map<Matrix6x4f,0,Eigen::Stride<6*stride(),stride()> >;
  using Map4d = Eigen::Map<Vector4d,0,Eigen::InnerStride<stride()> >;

}

template<typename M3x4, typename M6x4>
void fillHitsAndHitsCov(M3x4 & hits, M6x4 & hits_ge) {
  hits << 1.98645, 4.72598, 7.65632, 11.3151,
          2.18002, 4.88864, 7.75845, 11.3134,
          2.46338, 6.99838,  11.808,  17.793;
  hits_ge.col(0)[0] = 7.14652e-06;
  hits_ge.col(1)[0] = 2.15789e-06;
  hits_ge.col(2)[0] = 1.63328e-06;
  hits_ge.col(3)[0] = 6.27919e-06;
  hits_ge.col(0)[2] = 6.10348e-06;
  hits_ge.col(1)[2] = 2.08211e-06;
  hits_ge.col(2)[2] = 1.61672e-06;
  hits_ge.col(3)[2] = 6.28081e-06;
  hits_ge.col(0)[5] = 5.184e-05;
  hits_ge.col(1)[5] = 1.444e-05;
  hits_ge.col(2)[5] = 6.25e-06;
  hits_ge.col(3)[5] = 3.136e-05;
  hits_ge.col(0)[1] = -5.60077e-06;
  hits_ge.col(1)[1] = -1.11936e-06;
  hits_ge.col(2)[1] = -6.24945e-07;
  hits_ge.col(3)[1] = -5.28e-06;
}

void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd<4> hits;
  Rfit::Matrix6x4f hits_ge = MatrixXf::Zero(6,4);

  fillHitsAndHitsCov(hits, hits_ge);

  std::cout << "sizes " << sizeof(hits) << ' ' << sizeof(hits_ge)
	    << ' ' << sizeof(Vector4d)<< std::endl;
  
  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

  // FAST_FIT_CPU
  Vector4d fast_fit_results; Rfit::Fast_fit(hits, fast_fit_results);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;


  // CIRCLE_FIT CPU
  constexpr uint32_t N = Rfit::Map3x4d::ColsAtCompileTime;
  constexpr auto n = N;
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov =  MatrixXd::Zero(2 * n, 2 * n);
  Rfit::loadCovariance2D(hits_ge,hits_cov);
  Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
      hits_cov,
      fast_fit_results, rad, B, true);
  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;

  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
}

int main (int argc, char * argv[]) {
  testFit();
  return 0;
}

