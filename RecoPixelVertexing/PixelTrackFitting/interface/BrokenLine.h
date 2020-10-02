#ifndef RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
#define RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h

#include <Eigen/Eigenvalues>

#include "RecoPixelVertexing/PixelTrackFitting/interface/FitUtils.h"

namespace BrokenLine {

  //!< Karimäki's parameters: (phi, d, k=1/R)
  /*!< covariance matrix: \n
    |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
    |cov(phi, d )|cov( d , d )|cov( k , d )| \n
    |cov(phi, k )|cov( d , k )|cov( k , k )|
  */
  using karimaki_circle_fit = Rfit::circle_fit;

  /*!
    \brief data needed for the Broken Line fit procedure.
  */
  template <int N>
  struct PreparedBrokenLineData {
    int q;                      //!< particle charge
    Rfit::Matrix2xNd<N> radii;  //!< xy data in the system in which the pre-fitted center is the origin
    Rfit::VectorNd<N> s;        //!< total distance traveled in the transverse plane
                                //   starting from the pre-fitted closest approach
    Rfit::VectorNd<N> S;        //!< total distance traveled (three-dimensional)
    Rfit::VectorNd<N> Z;        //!< orthogonal coordinate to the pre-fitted line in the sz plane
    Rfit::VectorNd<N> VarBeta;  //!< kink angles in the SZ plane
  };

  /*!
    \brief Computes the Coulomb multiple scattering variance of the planar angle.
    
    \param length length of the track in the material.
    \param B magnetic field in Gev/cm/c.
    \param R radius of curvature (needed to evaluate p).
    \param Layer denotes which of the four layers of the detector is the endpoint of the multiple scattered track. For example, if Layer=3, then the particle has just gone through the material between the second and the third layer.
    
    \todo add another Layer variable to identify also the start point of the track, so if there are missing hits or multiple hits, the part of the detector that the particle has traversed can be exactly identified.
    
    \warning the formula used here assumes beta=1, and so neglects the dependence of theta_0 on the mass of the particle at fixed momentum.
    
    \return the variance of the planar angle ((theta_0)^2 /3).
  */
  __host__ __device__ inline double MultScatt(
      const double& length, const double B, const double R, int Layer, double slope) {
    // limit R to 20GeV...
    auto pt2 = std::min(20., B * R);
    pt2 *= pt2;
    constexpr double XXI_0 = 0.06 / 16.;  //!< inverse of radiation length of the material in cm
    //if(Layer==1) XXI_0=0.06/16.;
    // else XXI_0=0.06/16.;
    //XX_0*=1;
    constexpr double geometry_factor =
        0.7;  //!< number between 1/3 (uniform material) and 1 (thin scatterer) to be manually tuned
    constexpr double fact = geometry_factor * Rfit::sqr(13.6 / 1000.);
    return fact / (pt2 * (1. + Rfit::sqr(slope))) * (std::abs(length) * XXI_0) *
           Rfit::sqr(1. + 0.038 * log(std::abs(length) * XXI_0));
  }

  /*!
    \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.
    
    \param slope tangent of the angle of rotation.
    
    \return 2D rotation matrix.
  */
  __host__ __device__ inline Rfit::Matrix2d RotationMatrix(double slope) {
    Rfit::Matrix2d Rot;
    Rot(0, 0) = 1. / sqrt(1. + Rfit::sqr(slope));
    Rot(0, 1) = slope * Rot(0, 0);
    Rot(1, 0) = -Rot(0, 1);
    Rot(1, 1) = Rot(0, 0);
    return Rot;
  }

  /*!
    \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a translation of the coordinate system, such that the old origin has coordinates (x0,y0) in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.
    
    \param circle circle fit in the old coordinate system.
    \param x0 x coordinate of the translation vector.
    \param y0 y coordinate of the translation vector.
    \param jacobian passed by reference in order to save stack.
  */
  __host__ __device__ inline void TranslateKarimaki(karimaki_circle_fit& circle,
                                                    double x0,
                                                    double y0,
                                                    Rfit::Matrix3d& jacobian) {
    double A, U, BB, C, DO, DP, uu, xi, v, mu, lambda, zeta;
    DP = x0 * cos(circle.par(0)) + y0 * sin(circle.par(0));
    DO = x0 * sin(circle.par(0)) - y0 * cos(circle.par(0)) + circle.par(1);
    uu = 1 + circle.par(2) * circle.par(1);
    C = -circle.par(2) * y0 + uu * cos(circle.par(0));
    BB = circle.par(2) * x0 + uu * sin(circle.par(0));
    A = 2. * DO + circle.par(2) * (Rfit::sqr(DO) + Rfit::sqr(DP));
    U = sqrt(1. + circle.par(2) * A);
    xi = 1. / (Rfit::sqr(BB) + Rfit::sqr(C));
    v = 1. + circle.par(2) * DO;
    lambda = (0.5 * A) / (U * Rfit::sqr(1. + U));
    mu = 1. / (U * (1. + U)) + circle.par(2) * lambda;
    zeta = Rfit::sqr(DO) + Rfit::sqr(DP);

    jacobian << xi * uu * v, -xi * Rfit::sqr(circle.par(2)) * DP, xi * DP, 2. * mu * uu * DP, 2. * mu * v,
        mu * zeta - lambda * A, 0, 0, 1.;

    circle.par(0) = atan2(BB, C);
    circle.par(1) = A / (1 + U);
    // circle.par(2)=circle.par(2);

    circle.cov = jacobian * circle.cov * jacobian.transpose();
  }

  /*!
    \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.
    
    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
  */
  template <typename M3xN, typename V4, int N>
  __host__ __device__ inline void prepareBrokenLineData(const M3xN& hits,
                                                        const V4& fast_fit,
                                                        const double B,
                                                        PreparedBrokenLineData<N>& results) {
    constexpr auto n = N;
    u_int i;
    Rfit::Vector2d d;
    Rfit::Vector2d e;

    d = hits.block(0, 1, 2, 1) - hits.block(0, 0, 2, 1);
    e = hits.block(0, n - 1, 2, 1) - hits.block(0, n - 2, 2, 1);
    results.q = Rfit::cross2D(d, e) > 0 ? -1 : 1;

    const double slope = -results.q / fast_fit(3);

    Rfit::Matrix2d R = RotationMatrix(slope);

    // calculate radii and s
    results.radii = hits.block(0, 0, 2, n) - fast_fit.head(2) * Rfit::MatrixXd::Constant(1, n, 1);
    e = -fast_fit(2) * fast_fit.head(2) / fast_fit.head(2).norm();
    for (i = 0; i < n; i++) {
      d = results.radii.block(0, i, 2, 1);
      results.s(i) = results.q * fast_fit(2) * atan2(Rfit::cross2D(d, e), d.dot(e));  // calculates the arc length
    }
    Rfit::VectorNd<N> z = hits.block(2, 0, 1, n).transpose();

    //calculate S and Z
    Rfit::Matrix2xNd<N> pointsSZ = Rfit::Matrix2xNd<N>::Zero();
    for (i = 0; i < n; i++) {
      pointsSZ(0, i) = results.s(i);
      pointsSZ(1, i) = z(i);
      pointsSZ.block(0, i, 2, 1) = R * pointsSZ.block(0, i, 2, 1);
    }
    results.S = pointsSZ.block(0, 0, 1, n).transpose();
    results.Z = pointsSZ.block(1, 0, 1, n).transpose();

    //calculate VarBeta
    results.VarBeta(0) = results.VarBeta(n - 1) = 0;
    for (i = 1; i < n - 1; i++) {
      results.VarBeta(i) = MultScatt(results.S(i + 1) - results.S(i), B, fast_fit(2), i + 2, slope) +
                           MultScatt(results.S(i) - results.S(i - 1), B, fast_fit(2), i + 1, slope);
    }
  }

  /*!
    \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. This is the whole matrix in the case of the line fit and the main n-by-n block in the case of the circle fit.
    
    \param w weights of the first part of the cost function, the one with the measurements and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
    \param S total distance traveled by the particle from the pre-fitted closest approach.
    \param VarBeta kink angles' variance.
    
    \return the n-by-n matrix of the linear system
  */
  template <int N>
  __host__ __device__ inline Rfit::MatrixNd<N> MatrixC_u(const Rfit::VectorNd<N>& w,
                                                         const Rfit::VectorNd<N>& S,
                                                         const Rfit::VectorNd<N>& VarBeta) {
    constexpr u_int n = N;
    u_int i;

    Rfit::MatrixNd<N> C_U = Rfit::MatrixNd<N>::Zero();
    for (i = 0; i < n; i++) {
      C_U(i, i) = w(i);
      if (i > 1)
        C_U(i, i) += 1. / (VarBeta(i - 1) * Rfit::sqr(S(i) - S(i - 1)));
      if (i > 0 && i < n - 1)
        C_U(i, i) += (1. / VarBeta(i)) * Rfit::sqr((S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))));
      if (i < n - 2)
        C_U(i, i) += 1. / (VarBeta(i + 1) * Rfit::sqr(S(i + 1) - S(i)));

      if (i > 0 && i < n - 1)
        C_U(i, i + 1) =
            1. / (VarBeta(i) * (S(i + 1) - S(i))) * (-(S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))));
      if (i < n - 2)
        C_U(i, i + 1) += 1. / (VarBeta(i + 1) * (S(i + 1) - S(i))) *
                         (-(S(i + 2) - S(i)) / ((S(i + 2) - S(i + 1)) * (S(i + 1) - S(i))));

      if (i < n - 2)
        C_U(i, i + 2) = 1. / (VarBeta(i + 1) * (S(i + 2) - S(i + 1)) * (S(i + 1) - S(i)));

      C_U(i, i) *= 0.5;
    }
    return C_U + C_U.transpose();
  }

  /*!
    \brief A very fast helix fit.
    
    \param hits the measured hits.
    
    \return (X0,Y0,R,tan(theta)).
    
    \warning sign of theta is (intentionally, for now) mistaken for negative charges.
  */

  template <typename M3xN, typename V4>
  __host__ __device__ inline void BL_Fast_fit(const M3xN& hits, V4& result) {
    constexpr uint32_t N = M3xN::ColsAtCompileTime;
    constexpr auto n = N;  // get the number of hits

    const Rfit::Vector2d a = hits.block(0, n / 2, 2, 1) - hits.block(0, 0, 2, 1);
    const Rfit::Vector2d b = hits.block(0, n - 1, 2, 1) - hits.block(0, n / 2, 2, 1);
    const Rfit::Vector2d c = hits.block(0, 0, 2, 1) - hits.block(0, n - 1, 2, 1);

    auto tmp = 0.5 / Rfit::cross2D(c, a);
    result(0) = hits(0, 0) - (a(1) * c.squaredNorm() + c(1) * a.squaredNorm()) * tmp;
    result(1) = hits(1, 0) + (a(0) * c.squaredNorm() + c(0) * a.squaredNorm()) * tmp;
    // check Wikipedia for these formulas

    result(2) = sqrt(a.squaredNorm() * b.squaredNorm() * c.squaredNorm()) / (2. * std::abs(Rfit::cross2D(b, a)));
    // Using Math Olympiad's formula R=abc/(4A)

    const Rfit::Vector2d d = hits.block(0, 0, 2, 1) - result.head(2);
    const Rfit::Vector2d e = hits.block(0, n - 1, 2, 1) - result.head(2);

    result(3) = result(2) * atan2(Rfit::cross2D(d, e), d.dot(e)) / (hits(2, n - 1) - hits(2, 0));
    // ds/dz slope between last and first point
  }

  /*!
    \brief Performs the Broken Line fit in the curved track case (that is, the fit parameters are the interceptions u and the curvature correction \Delta\kappa).
    
    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param circle_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (phi, d, k); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \details The function implements the steps 2 and 3 of the Broken Line fit with the curvature correction.\n
    The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and \Delta\kappa and their covariance matrix.
    The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
  */
  template <typename M3xN, typename M6xN, typename V4, int N>
  __host__ __device__ inline void BL_Circle_fit(const M3xN& hits,
                                                const M6xN& hits_ge,
                                                const V4& fast_fit,
                                                const double B,
                                                PreparedBrokenLineData<N>& data,
                                                karimaki_circle_fit& circle_results) {
    constexpr u_int n = N;
    u_int i;

    circle_results.q = data.q;
    auto& radii = data.radii;
    const auto& s = data.s;
    const auto& S = data.S;
    auto& Z = data.Z;
    auto& VarBeta = data.VarBeta;
    const double slope = -circle_results.q / fast_fit(3);
    VarBeta *= 1. + Rfit::sqr(slope);  // the kink angles are projected!

    for (i = 0; i < n; i++) {
      Z(i) = radii.block(0, i, 2, 1).norm() - fast_fit(2);
    }

    Rfit::Matrix2d V;     // covariance matrix
    Rfit::VectorNd<N> w;  // weights
    Rfit::Matrix2d RR;    // rotation matrix point by point
    //double Slope; // slope of the circle point by point
    for (i = 0; i < n; i++) {
      V(0, 0) = hits_ge.col(i)[0];            // x errors
      V(0, 1) = V(1, 0) = hits_ge.col(i)[1];  // cov_xy
      V(1, 1) = hits_ge.col(i)[2];            // y errors
      //Slope=-radii(0,i)/radii(1,i);
      RR = RotationMatrix(-radii(0, i) / radii(1, i));
      w(i) = 1. / ((RR * V * RR.transpose())(1, 1));  // compute the orthogonal weight point by point
    }

    Rfit::VectorNplusONEd<N> r_u;
    r_u(n) = 0;
    for (i = 0; i < n; i++) {
      r_u(i) = w(i) * Z(i);
    }

    Rfit::MatrixNplusONEd<N> C_U;
    C_U.block(0, 0, n, n) = MatrixC_u(w, s, VarBeta);
    C_U(n, n) = 0;
    //add the border to the C_u matrix
    for (i = 0; i < n; i++) {
      C_U(i, n) = 0;
      if (i > 0 && i < n - 1) {
        C_U(i, n) +=
            -(s(i + 1) - s(i - 1)) * (s(i + 1) - s(i - 1)) / (2. * VarBeta(i) * (s(i + 1) - s(i)) * (s(i) - s(i - 1)));
      }
      if (i > 1) {
        C_U(i, n) += (s(i) - s(i - 2)) / (2. * VarBeta(i - 1) * (s(i) - s(i - 1)));
      }
      if (i < n - 2) {
        C_U(i, n) += (s(i + 2) - s(i)) / (2. * VarBeta(i + 1) * (s(i + 1) - s(i)));
      }
      C_U(n, i) = C_U(i, n);
      if (i > 0 && i < n - 1)
        C_U(n, n) += Rfit::sqr(s(i + 1) - s(i - 1)) / (4. * VarBeta(i));
    }

#ifdef CPP_DUMP
    std::cout << "CU5\n" << C_U << std::endl;
#endif
    Rfit::MatrixNplusONEd<N> I;
    math::cholesky::invert(C_U, I);
    // Rfit::MatrixNplusONEd<N> I = C_U.inverse();
#ifdef CPP_DUMP
    std::cout << "I5\n" << I << std::endl;
#endif

    Rfit::VectorNplusONEd<N> u = I * r_u;  // obtain the fitted parameters by solving the linear system

    // compute (phi, d_ca, k) in the system in which the midpoint of the first two corrected hits is the origin...

    radii.block(0, 0, 2, 1) /= radii.block(0, 0, 2, 1).norm();
    radii.block(0, 1, 2, 1) /= radii.block(0, 1, 2, 1).norm();

    Rfit::Vector2d d = hits.block(0, 0, 2, 1) + (-Z(0) + u(0)) * radii.block(0, 0, 2, 1);
    Rfit::Vector2d e = hits.block(0, 1, 2, 1) + (-Z(1) + u(1)) * radii.block(0, 1, 2, 1);

    circle_results.par << atan2((e - d)(1), (e - d)(0)),
        -circle_results.q * (fast_fit(2) - sqrt(Rfit::sqr(fast_fit(2)) - 0.25 * (e - d).squaredNorm())),
        circle_results.q * (1. / fast_fit(2) + u(n));

    assert(circle_results.q * circle_results.par(1) <= 0);

    Rfit::Vector2d eMinusd = e - d;
    double tmp1 = eMinusd.squaredNorm();

    Rfit::Matrix3d jacobian;
    jacobian << (radii(1, 0) * eMinusd(0) - eMinusd(1) * radii(0, 0)) / tmp1,
        (radii(1, 1) * eMinusd(0) - eMinusd(1) * radii(0, 1)) / tmp1, 0,
        (circle_results.q / 2) * (eMinusd(0) * radii(0, 0) + eMinusd(1) * radii(1, 0)) /
            sqrt(Rfit::sqr(2 * fast_fit(2)) - tmp1),
        (circle_results.q / 2) * (eMinusd(0) * radii(0, 1) + eMinusd(1) * radii(1, 1)) /
            sqrt(Rfit::sqr(2 * fast_fit(2)) - tmp1),
        0, 0, 0, circle_results.q;

    circle_results.cov << I(0, 0), I(0, 1), I(0, n), I(1, 0), I(1, 1), I(1, n), I(n, 0), I(n, 1), I(n, n);

    circle_results.cov = jacobian * circle_results.cov * jacobian.transpose();

    //...Translate in the system in which the first corrected hit is the origin, adding the m.s. correction...

    TranslateKarimaki(circle_results, 0.5 * (e - d)(0), 0.5 * (e - d)(1), jacobian);
    circle_results.cov(0, 0) += (1 + Rfit::sqr(slope)) * MultScatt(S(1) - S(0), B, fast_fit(2), 2, slope);

    //...And translate back to the original system

    TranslateKarimaki(circle_results, d(0), d(1), jacobian);

    // compute chi2
    circle_results.chi2 = 0;
    for (i = 0; i < n; i++) {
      circle_results.chi2 += w(i) * Rfit::sqr(Z(i) - u(i));
      if (i > 0 && i < n - 1)
        circle_results.chi2 += Rfit::sqr(u(i - 1) / (s(i) - s(i - 1)) -
                                         u(i) * (s(i + 1) - s(i - 1)) / ((s(i + 1) - s(i)) * (s(i) - s(i - 1))) +
                                         u(i + 1) / (s(i + 1) - s(i)) + (s(i + 1) - s(i - 1)) * u(n) / 2) /
                               VarBeta(i);
    }

    // assert(circle_results.chi2>=0);
  }

  /*!
    \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
    
    \param hits hits coordinates.
    \param hits_cov hits covariance matrix.
    \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
    \param B magnetic field in Gev/cm/c.
    \param data PreparedBrokenLineData.
    \param line_results struct to be filled with the results in this form:
    -par parameter of the line in this form: (cot(theta), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2 value of the cost function in the minimum.
    
    \details The function implements the steps 2 and 3 of the Broken Line fit without the curvature correction.\n
    The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and their covariance matrix.
    The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
  */
  template <typename V4, typename M6xN, int N>
  __host__ __device__ inline void BL_Line_fit(const M6xN& hits_ge,
                                              const V4& fast_fit,
                                              const double B,
                                              const PreparedBrokenLineData<N>& data,
                                              Rfit::line_fit& line_results) {
    constexpr u_int n = N;
    u_int i;

    const auto& radii = data.radii;
    const auto& S = data.S;
    const auto& Z = data.Z;
    const auto& VarBeta = data.VarBeta;

    const double slope = -data.q / fast_fit(3);
    Rfit::Matrix2d R = RotationMatrix(slope);

    Rfit::Matrix3d V = Rfit::Matrix3d::Zero();                 // covariance matrix XYZ
    Rfit::Matrix2x3d JacobXYZtosZ = Rfit::Matrix2x3d::Zero();  // jacobian for computation of the error on s (xyz -> sz)
    Rfit::VectorNd<N> w = Rfit::VectorNd<N>::Zero();
    for (i = 0; i < n; i++) {
      V(0, 0) = hits_ge.col(i)[0];            // x errors
      V(0, 1) = V(1, 0) = hits_ge.col(i)[1];  // cov_xy
      V(0, 2) = V(2, 0) = hits_ge.col(i)[3];  // cov_xz
      V(1, 1) = hits_ge.col(i)[2];            // y errors
      V(2, 1) = V(1, 2) = hits_ge.col(i)[4];  // cov_yz
      V(2, 2) = hits_ge.col(i)[5];            // z errors
      auto tmp = 1. / radii.block(0, i, 2, 1).norm();
      JacobXYZtosZ(0, 0) = radii(1, i) * tmp;
      JacobXYZtosZ(0, 1) = -radii(0, i) * tmp;
      JacobXYZtosZ(1, 2) = 1.;
      w(i) = 1. / ((R * JacobXYZtosZ * V * JacobXYZtosZ.transpose() * R.transpose())(
                      1, 1));  // compute the orthogonal weight point by point
    }

    Rfit::VectorNd<N> r_u;
    for (i = 0; i < n; i++) {
      r_u(i) = w(i) * Z(i);
    }
#ifdef CPP_DUMP
    std::cout << "CU4\n" << MatrixC_u(w, S, VarBeta) << std::endl;
#endif
    Rfit::MatrixNd<N> I;
    math::cholesky::invert(MatrixC_u(w, S, VarBeta), I);
    //    Rfit::MatrixNd<N> I=MatrixC_u(w,S,VarBeta).inverse();
#ifdef CPP_DUMP
    std::cout << "I4\n" << I << std::endl;
#endif

    Rfit::VectorNd<N> u = I * r_u;  // obtain the fitted parameters by solving the linear system

    // line parameters in the system in which the first hit is the origin and with axis along SZ
    line_results.par << (u(1) - u(0)) / (S(1) - S(0)), u(0);
    auto idiff = 1. / (S(1) - S(0));
    line_results.cov << (I(0, 0) - 2 * I(0, 1) + I(1, 1)) * Rfit::sqr(idiff) +
                            MultScatt(S(1) - S(0), B, fast_fit(2), 2, slope),
        (I(0, 1) - I(0, 0)) * idiff, (I(0, 1) - I(0, 0)) * idiff, I(0, 0);

    // translate to the original SZ system
    Rfit::Matrix2d jacobian;
    jacobian(0, 0) = 1.;
    jacobian(0, 1) = 0;
    jacobian(1, 0) = -S(0);
    jacobian(1, 1) = 1.;
    line_results.par(1) += -line_results.par(0) * S(0);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

    // rotate to the original sz system
    auto tmp = R(0, 0) - line_results.par(0) * R(0, 1);
    jacobian(1, 1) = 1. / tmp;
    jacobian(0, 0) = jacobian(1, 1) * jacobian(1, 1);
    jacobian(0, 1) = 0;
    jacobian(1, 0) = line_results.par(1) * R(0, 1) * jacobian(0, 0);
    line_results.par(1) = line_results.par(1) * jacobian(1, 1);
    line_results.par(0) = (R(0, 1) + line_results.par(0) * R(0, 0)) * jacobian(1, 1);
    line_results.cov = jacobian * line_results.cov * jacobian.transpose();

    // compute chi2
    line_results.chi2 = 0;
    for (i = 0; i < n; i++) {
      line_results.chi2 += w(i) * Rfit::sqr(Z(i) - u(i));
      if (i > 0 && i < n - 1)
        line_results.chi2 += Rfit::sqr(u(i - 1) / (S(i) - S(i - 1)) -
                                       u(i) * (S(i + 1) - S(i - 1)) / ((S(i + 1) - S(i)) * (S(i) - S(i - 1))) +
                                       u(i + 1) / (S(i + 1) - S(i))) /
                             VarBeta(i);
    }

    // assert(line_results.chi2>=0);
  }

  /*!
    \brief Helix fit by three step:
    -fast pre-fit (see Fast_fit() for further info); \n
    -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
    -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
    Points must be passed ordered (from inner to outer layer).
    
    \param hits Matrix3xNd hits coordinates in this form: \n
    |x1|x2|x3|...|xn| \n
    |y1|y2|y3|...|yn| \n
    |z1|z2|z3|...|zn|
    \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
    |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
    |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
    |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
    |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
    .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
    |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
    |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
    |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
    |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
    .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
    |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
    |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
    |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
    |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
    \param B magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.
    
    \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.
    
    \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.
    
    \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
  */
  template <int N>
  inline Rfit::helix_fit BL_Helix_fit(const Rfit::Matrix3xNd<N>& hits,
                                      const Eigen::Matrix<float, 6, 4>& hits_ge,
                                      const double B) {
    Rfit::helix_fit helix;
    Rfit::Vector4d fast_fit;
    BL_Fast_fit(hits, fast_fit);

    PreparedBrokenLineData<N> data;
    karimaki_circle_fit circle;
    Rfit::line_fit line;
    Rfit::Matrix3d jacobian;

    prepareBrokenLineData(hits, fast_fit, B, data);
    BL_Line_fit(hits_ge, fast_fit, B, data, line);
    BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle);

    // the circle fit gives k, but here we want p_t, so let's change the parameter and the covariance matrix
    jacobian << 1., 0, 0, 0, 1., 0, 0, 0, -std::abs(circle.par(2)) * B / (Rfit::sqr(circle.par(2)) * circle.par(2));
    circle.par(2) = B / std::abs(circle.par(2));
    circle.cov = jacobian * circle.cov * jacobian.transpose();

    helix.par << circle.par, line.par;
    helix.cov = Rfit::MatrixXd::Zero(5, 5);
    helix.cov.block(0, 0, 3, 3) = circle.cov;
    helix.cov.block(3, 3, 2, 2) = line.cov;
    helix.q = circle.q;
    helix.chi2_circle = circle.chi2;
    helix.chi2_line = line.chi2;

    return helix;
  }

}  // namespace BrokenLine

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
