#ifndef RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h
#define RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h

#include "DataFormats/Math/interface/choleskyInversion.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"

namespace Rfit {

  constexpr double d = 1.e-4;  //!< used in numerical derivative (J2 in Circle_fit())

  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  template <int N>
  using MatrixNd = Eigen::Matrix<double, N, N>;
  template <int N>
  using MatrixNplusONEd = Eigen::Matrix<double, N + 1, N + 1>;
  template <int N>
  using ArrayNd = Eigen::Array<double, N, N>;
  template <int N>
  using Matrix2Nd = Eigen::Matrix<double, 2 * N, 2 * N>;
  template <int N>
  using Matrix3Nd = Eigen::Matrix<double, 3 * N, 3 * N>;
  template <int N>
  using Matrix2xNd = Eigen::Matrix<double, 2, N>;
  template <int N>
  using Array2xNd = Eigen::Array<double, 2, N>;
  template <int N>
  using MatrixNx3d = Eigen::Matrix<double, N, 3>;
  template <int N>
  using MatrixNx5d = Eigen::Matrix<double, N, 5>;
  template <int N>
  using VectorNd = Eigen::Matrix<double, N, 1>;
  template <int N>
  using VectorNplusONEd = Eigen::Matrix<double, N + 1, 1>;
  template <int N>
  using Vector2Nd = Eigen::Matrix<double, 2 * N, 1>;
  template <int N>
  using Vector3Nd = Eigen::Matrix<double, 3 * N, 1>;
  template <int N>
  using RowVectorNd = Eigen::Matrix<double, 1, 1, N>;
  template <int N>
  using RowVector2Nd = Eigen::Matrix<double, 1, 2 * N>;

  using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

  using Matrix3f = Eigen::Matrix3f;
  using Vector3f = Eigen::Vector3f;
  using Vector4f = Eigen::Vector4f;
  using Vector6f = Eigen::Matrix<double, 6, 1>;

  using u_int = unsigned int;

  template <class C>
  __host__ __device__ void printIt(C* m, const char* prefix = "") {
#ifdef RFIT_DEBUG
    for (u_int r = 0; r < m->rows(); ++r) {
      for (u_int c = 0; c < m->cols(); ++c) {
        printf("%s Matrix(%d,%d) = %g\n", prefix, r, c, (*m)(r, c));
      }
    }
#endif
  }

  /*!
    \brief raise to square.
  */
  template <typename T>
  constexpr T sqr(const T a) {
    return a * a;
  }

  /*!
    \brief Compute cross product of two 2D vector (assuming z component 0),
    returning z component of the result.
    \param a first 2D vector in the product.
    \param b second 2D vector in the product.
    \return z component of the cross product.
  */

  __host__ __device__ inline double cross2D(const Vector2d& a, const Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
  }

  /*!
   *  load error in CMSSW format to our formalism
   *  
   */
  template <typename M6xNf, typename M2Nd>
  __host__ __device__ void loadCovariance2D(M6xNf const& ge, M2Nd& hits_cov) {
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i = 0; i < hits_in_fit; ++i) {
      auto ge_idx = 0;
      auto j = 0;
      auto l = 0;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 2;
      j = 1;
      l = 1;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 1;
      j = 1;
      l = 0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
          ge.col(i)[ge_idx];
    }
  }

  template <typename M6xNf, typename M3xNd>
  __host__ __device__ void loadCovariance(M6xNf const& ge, M3xNd& hits_cov) {
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i = 0; i < hits_in_fit; ++i) {
      auto ge_idx = 0;
      auto j = 0;
      auto l = 0;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 2;
      j = 1;
      l = 1;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 5;
      j = 2;
      l = 2;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 1;
      j = 1;
      l = 0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
          ge.col(i)[ge_idx];
      ge_idx = 3;
      j = 2;
      l = 0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
          ge.col(i)[ge_idx];
      ge_idx = 4;
      j = 2;
      l = 1;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) = hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) =
          ge.col(i)[ge_idx];
    }
  }

  /*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,p_t) and
    consequently covariance matrix.
    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
    \param B magnetic field in Gev/cm/c unit.
    \param error flag for errors computation.
  */
  __host__ __device__ inline void par_uvrtopak(circle_fit& circle, const double B, const bool error) {
    Vector3d par_pak;
    const double temp0 = circle.par.head(2).squaredNorm();
    const double temp1 = sqrt(temp0);
    par_pak << atan2(circle.q * circle.par(0), -circle.q * circle.par(1)), circle.q * (temp1 - circle.par(2)),
        circle.par(2) * B;
    if (error) {
      const double temp2 = sqr(circle.par(0)) * 1. / temp0;
      const double temp3 = 1. / temp1 * circle.q;
      Matrix3d J4;
      J4 << -circle.par(1) * temp2 * 1. / sqr(circle.par(0)), temp2 * 1. / circle.par(0), 0., circle.par(0) * temp3,
          circle.par(1) * temp3, -circle.q, 0., 0., B;
      circle.cov = J4 * circle.cov * J4.transpose();
    }
    circle.par = par_pak;
  }

  /*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,q/R) and
    consequently covariance matrix.
    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
  */
  __host__ __device__ inline void fromCircleToPerigee(circle_fit& circle) {
    Vector3d par_pak;
    const double temp0 = circle.par.head(2).squaredNorm();
    const double temp1 = sqrt(temp0);
    par_pak << atan2(circle.q * circle.par(0), -circle.q * circle.par(1)), circle.q * (temp1 - circle.par(2)),
        circle.q / circle.par(2);

    const double temp2 = sqr(circle.par(0)) * 1. / temp0;
    const double temp3 = 1. / temp1 * circle.q;
    Matrix3d J4;
    J4 << -circle.par(1) * temp2 * 1. / sqr(circle.par(0)), temp2 * 1. / circle.par(0), 0., circle.par(0) * temp3,
        circle.par(1) * temp3, -circle.q, 0., 0., -circle.q / (circle.par(2) * circle.par(2));
    circle.cov = J4 * circle.cov * J4.transpose();

    circle.par = par_pak;
  }

  // transformation between the "perigee" to cmssw localcoord frame
  // the plane of the latter is the perigee plane...
  // from   //!<(phi,Tip,q/pt,cotan(theta)),Zip)
  // to q/p,dx/dz,dy/dz,x,z
  template <typename VI5, typename MI5, typename VO5, typename MO5>
  __host__ __device__ inline void transformToPerigeePlane(VI5 const& ip, MI5 const& icov, VO5& op, MO5& ocov) {
    auto sinTheta2 = 1. / (1. + ip(3) * ip(3));
    auto sinTheta = std::sqrt(sinTheta2);
    auto cosTheta = ip(3) * sinTheta;

    op(0) = sinTheta * ip(2);
    op(1) = 0.;
    op(2) = -ip(3);
    op(3) = ip(1);
    op(4) = -ip(4);

    Matrix5d J = Matrix5d::Zero();

    J(0, 2) = sinTheta;
    J(0, 3) = -sinTheta2 * cosTheta * ip(2);
    J(1, 0) = 1.;
    J(2, 3) = -1.;
    J(3, 1) = 1.;
    J(4, 4) = -1;

    ocov = J * icov * J.transpose();
  }

}  // namespace Rfit

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h
