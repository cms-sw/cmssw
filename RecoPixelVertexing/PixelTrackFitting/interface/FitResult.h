#ifndef RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h
#define RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace Rfit {

  using Vector2d = Eigen::Vector2d;
  using Vector3d = Eigen::Vector3d;
  using Vector4d = Eigen::Vector4d;
  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Matrix2d = Eigen::Matrix2d;
  using Matrix3d = Eigen::Matrix3d;
  using Matrix4d = Eigen::Matrix4d;
  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;  // used for inputs hits

  struct circle_fit {
    Vector3d par;  //!< parameter: (X0,Y0,R)
    Matrix3d cov;
    /*!< covariance matrix: \n
      |cov(X0,X0)|cov(Y0,X0)|cov( R,X0)| \n
      |cov(X0,Y0)|cov(Y0,Y0)|cov( R,Y0)| \n
      |cov(X0, R)|cov(Y0, R)|cov( R, R)|
    */
    int32_t q;  //!< particle charge
    float chi2;
  };

  struct line_fit {
    Vector2d par;  //!<(cotan(theta),Zip)
    Matrix2d cov;
    /*!<
      |cov(c_t,c_t)|cov(Zip,c_t)| \n
      |cov(c_t,Zip)|cov(Zip,Zip)|
    */
    double chi2;
  };

  struct helix_fit {
    Vector5d par;  //!<(phi,Tip,pt,cotan(theta)),Zip)
    Matrix5d cov;
    /*!< ()->cov() \n
      |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
      |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
      |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
      |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
      |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    */
    float chi2_circle;
    float chi2_line;
    //    Vector4d fast_fit;
    int32_t q;  //!< particle charge
  };            // __attribute__((aligned(16)));

}  // namespace Rfit
#endif
