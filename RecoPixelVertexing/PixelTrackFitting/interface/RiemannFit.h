#ifndef RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h
#define RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h

#include <cmath>

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#ifndef RFIT_DEBUG
#define RFIT_DEBUG 0
#endif  // RFIT_DEBUG

namespace Rfit
{
using namespace Eigen;

constexpr double d = 1.e-4;          //!< used in numerical derivative (J2 in Circle_fit())
constexpr unsigned int max_nop = 4;  //!< In order to avoid use of dynamic memory

using MatrixNd = Eigen::Matrix<double, Dynamic, Dynamic, 0, max_nop, max_nop>;
using ArrayNd = Eigen::Array<double, Dynamic, Dynamic, 0, max_nop, max_nop>;
using Matrix2Nd = Eigen::Matrix<double, Dynamic, Dynamic, 0, 2 * max_nop, 2 * max_nop>;
using Matrix3Nd = Eigen::Matrix<double, Dynamic, Dynamic, 0, 3 * max_nop, 3 * max_nop>;
using Matrix2xNd = Eigen::Matrix<double, 2, Dynamic, 0, 2, max_nop>;
using Array2xNd = Eigen::Array<double, 2, Dynamic, 0, 2, max_nop>;
using Matrix3xNd = Eigen::Matrix<double, 3, Dynamic, 0, 3, max_nop>;
using MatrixNx3d = Eigen::Matrix<double, Dynamic, 3, 0, max_nop, 3>;
using MatrixNx5d = Eigen::Matrix<double, Dynamic, 5, 0, max_nop, 5>;
using VectorNd = Eigen::Matrix<double, Dynamic, 1, 0, max_nop, 1>;
using Vector2Nd = Eigen::Matrix<double, Dynamic, 1, 0, 2 * max_nop, 1>;
using Vector3Nd = Eigen::Matrix<double, Dynamic, 1, 0, 3 * max_nop, 1>;
using RowVectorNd = Eigen::Matrix<double, 1, Dynamic, 1, 1, max_nop>;
using RowVector2Nd = Eigen::Matrix<double, 1, Dynamic, 1, 1, 2 * max_nop>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using u_int = unsigned int;

struct circle_fit
{
    Vector3d par;  //!< parameter: (X0,Y0,R)
    Matrix3d cov;
    /*!< covariance matrix: \n
      |cov(X0,X0)|cov(Y0,X0)|cov( R,X0)| \n
      |cov(X0,Y0)|cov(Y0,Y0)|cov( R,Y0)| \n
      |cov(X0, R)|cov(Y0, R)|cov( R, R)|
  */
    int64_t q;  //!< particle charge
    double chi2 = 0.0;
};

struct line_fit
{
    Vector2d par;  //!<(cotan(theta),Zip)
    Matrix2d cov;
    /*!<
      |cov(c_t,c_t)|cov(Zip,c_t)| \n
      |cov(c_t,Zip)|cov(Zip,Zip)|
  */
    double chi2 = 0.0;
};

struct helix_fit
{
    Vector5d par;  //!<(phi,Tip,pt,cotan(theta)),Zip)
    Matrix5d cov;
    /*!< ()->cov() \n
      |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
      |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
      |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
      |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
      |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
  */
    double chi2_circle = 0.0;
    double chi2_line = 0.0;
    Vector4d fast_fit;
    int64_t q;  //!< particle charge
                //  VectorXd time;  // TO FIX just for profiling
} __attribute__((aligned(16)));

template <class C>
__host__ __device__ void printIt(C* m, const char* prefix = "")
{
#if RFIT_DEBUG
    for (u_int r = 0; r < m->rows(); ++r)
    {
        for (u_int c = 0; c < m->cols(); ++c)
        {
            printf("%s Matrix(%d,%d) = %g\n", prefix, r, c, (*m)(r, c));
        }
    }
#endif
}

/*!
    \brief raise to square.
*/
template <typename T>
__host__ __device__ inline T sqr(const T a)
{
    return a * a;
}

/*!
    \brief Compute cross product of two 2D vector (assuming z component 0),
    returning z component of the result.

    \param a first 2D vector in the product.
    \param b second 2D vector in the product.

    \return z component of the cross product.
*/

__host__ __device__ inline double cross2D(const Vector2d& a, const Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}


__host__ __device__ inline void computeRadLenEff(const Vector4d& fast_fit,
                                                 const double B,
                                                 double & radlen_eff,
                                                 double & theta,
                                                 bool & in_forward) {
    double X_barrel = 0.015;
    double X_forward = 0.05;
    theta = atan(fast_fit(3));
    // atan returns values in [-pi/2, pi/2], we need [0, pi]
    theta = theta < 0. ? theta + M_PI : theta;
    radlen_eff = X_barrel / std::abs(sin(theta));
    in_forward = (theta <= 0.398 or theta >= 2.743);
    if (in_forward)
      radlen_eff = X_forward / std::abs(cos(theta));
    assert(radlen_eff > 0.);
    double p_t = fast_fit(2) * B;
    // We have also to correct the radiation lenght in the x-y plane. Since we
    // do not know the angle of incidence of the track at this point, we
    // arbitrarily set the correction proportional to the inverse of the
    // transerse momentum. The cut-off is at 1 Gev, set using a single Muon Pt
    // gun and verifying that, at that momentum, not additional correction is,
    // in fact, needed. This is an approximation.
    if (std::abs(p_t/1.) < 1.)
      radlen_eff /= std::abs(p_t/1.);
}

/*!
    \brief Compute the covariance matrix along cartesian S-Z of points due to
    multiple Coulomb scattering to be used in the line_fit, for the barrel
    and forward cases.

 */
__host__ __device__ inline MatrixNd Scatter_cov_line(Matrix2Nd& cov_sz,
                                                     const Vector4d& fast_fit,
                                                     VectorNd const& s_arcs,
                                                     VectorNd const& z_values,
                                                     const double B)
{
#if RFIT_DEBUG
    Rfit::printIt(&s_arcs, "Scatter_cov_line - s_arcs: ");
#endif
    u_int n = s_arcs.rows();
    double p_t = fast_fit(2) * B;
    double p_2 = p_t * p_t * (1. + 1. / (fast_fit(3) * fast_fit(3)));
    double radlen_eff = 0.;
    double theta = 0.;
    bool in_forward = false;
    computeRadLenEff(fast_fit, B, radlen_eff, theta, in_forward);

    const double sig2 = .000225 / p_2 * sqr(1 + 0.038 * log(radlen_eff)) * radlen_eff;
    for (u_int k = 0; k < n; ++k)
    {
        for (u_int l = k; l < n; ++l)
        {
            for (u_int i = 0; i < std::min(k, l); ++i)
            {
#if RFIT_DEBUG
              printf("Scatter_cov_line - B: %f\n", B);
              printf("Scatter_cov_line - radlen_eff: %f, p_t: %f, p2: %f\n", radlen_eff, p_t, p_2);
              printf("Scatter_cov_line - sig2:%f, theta: %f\n", sig2, theta);
              printf("Scatter_cov_line - Adding to element %d, %d value %f\n", n + k, n + l, (s_arcs(k) - s_arcs(i)) * (s_arcs(l) - s_arcs(i)) * sig2 / sqr(sqr(sin(theta))));
#endif
              if (in_forward) {
                cov_sz(k, l) += (z_values(k) - z_values(i)) * (z_values(l) - z_values(i)) * sig2 / sqr(sqr(cos(theta)));
                cov_sz(l, k) = cov_sz(k, l);
              } else {
                cov_sz(n + k, n + l) += (s_arcs(k) - s_arcs(i)) * (s_arcs(l) - s_arcs(i)) * sig2 / sqr(sqr(sin(theta)));
                cov_sz(n + l, n + k) = cov_sz(n + k, n + l);
              }
            }
        }
    }
#if RFIT_DEBUG
    Rfit::printIt(&cov_sz, "Scatter_cov_line - cov_sz: ");
#endif
    Matrix2Nd rot = MatrixXd::Zero(2 * n, 2 * n);
    for (u_int i = 0; i < n; ++i) {
      rot(i, i) = cos(theta);
      rot(n + i, n + i) = cos(theta);
      u_int j = (i + n);
      // Signs seem to be wrong for the off-diagonal element, but we are
      // inverting x-y in the input vector, since theta is the angle between
      // the z axis and the line, and we are putting the s values, which are Y,
      // in the first position. A simple sign flip will take care of it.
      rot(i, j) = i < j ? sin(theta) : -sin(theta);
    }

#if RFIT_DEBUG
    Rfit::printIt(&rot, "Scatter_cov_line - rot: ");
#endif

    Matrix2Nd tmp = rot*cov_sz*rot.transpose();
    // We are interested only in the errors in the rotated s -axis which, in
    // our formalism, are in the upper square matrix.
#if RFIT_DEBUG
    Rfit::printIt(&tmp, "Scatter_cov_line - tmp: ");
#endif
    return tmp.block(0, 0, n, n);
}

/*!
    \brief Compute the covariance matrix (in radial coordinates) of points in
    the transverse plane due to multiple Coulomb scattering.

    \param p2D 2D points in the transverse plane.
    \param fast_fit fast_fit Vector4d result of the previous pre-fit
    structured in this form:(X0, Y0, R, Tan(Theta))).
    \param B magnetic field use to compute p

    \return scatter_cov_rad errors due to multiple scattering.

    \warning input points must be ordered radially from the detector center
    (from inner layer to outer ones; points on the same layer must ordered too).
    \bug currently works only for points in the barrel.

    \details Only the tangential component is computed (the radial one is
    negligible).

 */
// X in input TO FIX
__host__ __device__ inline MatrixNd Scatter_cov_rad(const Matrix2xNd& p2D,
                                                    const Vector4d& fast_fit,
                                                    VectorNd const& rad,
                                                    double B)
{
    u_int n = p2D.cols();
    double p_t = fast_fit(2) * B;
    double p_2 = p_t * p_t * (1. + 1. / (fast_fit(3) * fast_fit(3)));
    double radlen_eff = 0.;
    double theta = 0.;
    bool in_forward = false;
    computeRadLenEff(fast_fit, B, radlen_eff, theta, in_forward);

    MatrixNd scatter_cov_rad = MatrixXd::Zero(n, n);
    const double sig2 = .000225 / p_2 * sqr(1 + 0.038 * log(radlen_eff)) * radlen_eff;
    for (u_int k = 0; k < n; ++k)
    {
        for (u_int l = k; l < n; ++l)
        {
            for (u_int i = 0; i < std::min(k, l); ++i)
            {
              if (in_forward) {
                scatter_cov_rad(k, l) += (rad(k) - rad(i)) * (rad(l) - rad(i)) * sig2 / sqr(cos(theta));
              } else {
                scatter_cov_rad(k, l) += (rad(k) - rad(i)) * (rad(l) - rad(i)) * sig2 / sqr(sin(theta));
              }
              scatter_cov_rad(l, k) = scatter_cov_rad(k, l);
            }
        }
    }
#if RFIT_DEBUG
    Rfit::printIt(&scatter_cov_rad, "Scatter_cov_rad - scatter_cov_rad: ");
#endif
    return scatter_cov_rad;
}

/*!
    \brief Transform covariance matrix from radial (only tangential component)
    to Cartesian coordinates (only transverse plane component).

    \param p2D 2D points in the transverse plane.
    \param cov_rad covariance matrix in radial coordinate.

    \return cov_cart covariance matrix in Cartesian coordinates.
*/

__host__ __device__ inline Matrix2Nd cov_radtocart(const Matrix2xNd& p2D,
                                                   const MatrixNd& cov_rad,
                                                   const VectorNd& rad)
{
#if RFIT_DEBUG
    printf("Address of p2D: %p\n", &p2D);
#endif
    printIt(&p2D, "cov_radtocart - p2D:");
    u_int n = p2D.cols();
    Matrix2Nd cov_cart = MatrixXd::Zero(2 * n, 2 * n);
    VectorNd rad_inv = rad.cwiseInverse();
    printIt(&rad_inv, "cov_radtocart - rad_inv:");
    for (u_int i = 0; i < n; ++i)
    {
        for (u_int j = i; j < n; ++j)
        {
            cov_cart(i, j) = cov_rad(i, j) * p2D(1, i) * rad_inv(i) * p2D(1, j) * rad_inv(j);
            cov_cart(i + n, j + n) = cov_rad(i, j) * p2D(0, i) * rad_inv(i) * p2D(0, j) * rad_inv(j);
            cov_cart(i, j + n) = -cov_rad(i, j) * p2D(1, i) * rad_inv(i) * p2D(0, j) * rad_inv(j);
            cov_cart(i + n, j) = -cov_rad(i, j) * p2D(0, i) * rad_inv(i) * p2D(1, j) * rad_inv(j);

            cov_cart(j, i) = cov_cart(i, j);
            cov_cart(j + n, i + n) = cov_cart(i + n, j + n);
            cov_cart(j + n, i) = cov_cart(i, j + n);
            cov_cart(j, i + n) = cov_cart(i + n, j);
        }
    }
    return cov_cart;
}

/*!
    \brief Transform covariance matrix from Cartesian coordinates (only
    transverse plane component) to radial coordinates (both radial and
    tangential component but only diagonal terms, correlation between different
    point are not managed).

    \param p2D 2D points in transverse plane.
    \param cov_cart covariance matrix in Cartesian coordinates.

    \return cov_rad covariance matrix in raidal coordinate.

    \warning correlation between different point are not computed.
*/
__host__ __device__ inline MatrixNd cov_carttorad(const Matrix2xNd& p2D,
                                                  const Matrix2Nd& cov_cart,
                                                  const VectorNd& rad)
{
    u_int n = p2D.cols();
    MatrixNd cov_rad = MatrixXd::Zero(n, n);
    const VectorNd rad_inv2 = rad.cwiseInverse().array().square();
    for (u_int i = 0; i < n; ++i)
    {
        //!< in case you have (0,0) to avoid dividing by 0 radius
        if (rad(i) < 1.e-4)
            cov_rad(i, i) = cov_cart(i, i);
        else
        {
            cov_rad(i, i) = rad_inv2(i) * (cov_cart(i, i) * sqr(p2D(1, i)) + cov_cart(i + n, i + n) * sqr(p2D(0, i)) - 2. * cov_cart(i, i + n) * p2D(0, i) * p2D(1, i));
        }
    }
    return cov_rad;
}

/*!
    \brief Transform covariance matrix from Cartesian coordinates (only
    transverse plane component) to coordinates system orthogonal to the
    pre-fitted circle in each point.
    Further information in attached documentation.

    \param p2D 2D points in transverse plane.
    \param cov_cart covariance matrix in Cartesian coordinates.
    \param fast_fit fast_fit Vector4d result of the previous pre-fit
    structured in this form:(X0, Y0, R, tan(theta))).

    \return cov_rad covariance matrix in the pre-fitted circle's
    orthogonal system.

*/

__host__ __device__ inline MatrixNd cov_carttorad_prefit(const Matrix2xNd& p2D, const Matrix2Nd& cov_cart,
                                                         const Vector4d& fast_fit,
                                                         const VectorNd& rad)
{
    u_int n = p2D.cols();
    MatrixNd cov_rad = MatrixXd::Zero(n, n);
    for (u_int i = 0; i < n; ++i)
    {
        //!< in case you have (0,0) to avoid dividing by 0 radius
        if (rad(i) < 1.e-4)
            cov_rad(i, i) = cov_cart(i, i);  // TO FIX
        else
        {
            Vector2d a = p2D.col(i);
            Vector2d b = p2D.col(i) - fast_fit.head(2);
            const double x2 = a.dot(b);
            const double y2 = cross2D(a, b);
            const double tan_c = -y2 / x2;
            const double tan_c2 = sqr(tan_c);
            cov_rad(i, i) = 1. / (1. + tan_c2) * (cov_cart(i, i) + cov_cart(i + n, i + n) * tan_c2 + 2 * cov_cart(i, i + n) * tan_c);
        }
    }
    return cov_rad;
}

/*!
    \brief Compute the points' weights' vector for the circle fit when multiple
    scattering is managed.
    Further information in attached documentation.

    \param cov_rad_inv covariance matrix inverse in radial coordinated
    (or, beter, pre-fitted circle's orthogonal system).

    \return weight VectorNd points' weights' vector.

    \bug I'm not sure this is the right way to compute the weights for non
    diagonal cov matrix. Further investigation needed.
*/

__host__ __device__ inline VectorNd Weight_circle(const MatrixNd& cov_rad_inv)
{
    return cov_rad_inv.colwise().sum().transpose();
}

/*!
    \brief Compute the points' weights' vector for the line fit (ODR).
    Results from a pre-fit is needed in order to take the orthogonal (to the
    line) component of the errors.

    \param x_err2 squared errors in the x axis.
    \param y_err2 squared errors in the y axis.
    \param tan_theta tangent of theta (angle between y axis and line).

    \return weight points' weights' vector for the line fit (ODR).
*/

__host__ __device__ inline VectorNd Weight_line(const ArrayNd& x_err2, const ArrayNd& y_err2, const double& tan_theta)
{
    return (1. + sqr(tan_theta)) * 1. / (x_err2 + y_err2 * sqr(tan_theta));
}

/*!
    \brief Find particle q considering the  sign of cross product between
    particles velocity (estimated by the first 2 hits) and the vector radius
    between the first hit and the center of the fitted circle.

    \param p2D 2D points in transverse plane.
    \param par_uvr result of the circle fit in this form: (X0,Y0,R).

    \return q int 1 or -1.
*/

__host__ __device__ inline int64_t Charge(const Matrix2xNd& p2D, const Vector3d& par_uvr)
{
    return ((p2D(0, 1) - p2D(0, 0)) * (par_uvr.y() - p2D(1, 0)) - (p2D(1, 1) - p2D(1, 0)) * (par_uvr.x() - p2D(0, 0)) > 0)? -1 : 1;
}

/*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,p_t) and
    consequently covariance matrix.

    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
    \param B magnetic field in Gev/cm/c unit.
    \param error flag for errors computation.
*/

__host__ __device__ inline void par_uvrtopak(circle_fit& circle, const double B, const bool error)
{
    Vector3d par_pak;
    const double temp0 = circle.par.head(2).squaredNorm();
    const double temp1 = sqrt(temp0);
    par_pak << atan2(circle.q * circle.par(0), -circle.q * circle.par(1)),
        circle.q * (temp1 - circle.par(2)), circle.par(2) * B;
    if (error)
    {
        const double temp2 = sqr(circle.par(0)) * 1. / temp0;
        const double temp3 = 1. / temp1 * circle.q;
        Matrix3d J4;
        J4 << -circle.par(1) * temp2 * 1. / sqr(circle.par(0)), temp2 * 1. / circle.par(0), 0., circle.par(0) * temp3, circle.par(1) * temp3, -circle.q, 0., 0., B;
        circle.cov = J4 * circle.cov * J4.transpose();
    }
    circle.par = par_pak;
}

/*!
    \brief Compute the error propagation to obtain the square errors in the
    x axis for the line fit. If errors have not been computed in the circle fit
    than an'approximation is made.
    Further information in attached documentation.

    \param V hits' covariance matrix.
    \param circle result of the previous circle fit (only the covariance matrix
    is needed) TO FIX
    \param J Jacobian of the transformation producing x values.
    \param error flag for error computation.

    \return x_err2 squared errors in the x axis.
*/

__host__ __device__ inline VectorNd X_err2(const Matrix3Nd& V, const circle_fit& circle, const MatrixNx5d& J,
                                           const bool error, u_int n)
{
    VectorNd x_err2(n);
    for (u_int i = 0; i < n; ++i)
    {
        Matrix5d Cov = MatrixXd::Zero(5, 5);
        if (error)
            Cov.block(0, 0, 3, 3) = circle.cov;
        Cov(3, 3) = V(i, i);
        Cov(4, 4) = V(i + n, i + n);
        Cov(3, 4) = Cov(4, 3) = V(i, i + n);
        Eigen::Matrix<double, 1, 1> tmp;
        tmp = J.row(i) * Cov * J.row(i).transpose().eval();
        x_err2(i) = tmp(0, 0);
    }
    return x_err2;
}

/*!
    \brief Compute the eigenvector associated to the minimum eigenvalue.

    \param A the Matrix you want to know eigenvector and eigenvalue.
    \param chi2 the double were the chi2-related quantity will be stored.

    \return the eigenvector associated to the minimum eigenvalue.

    \warning double precision is needed for a correct assessment of chi2.

    \details The minimus eigenvalue is related to chi2.
    We exploit the fact that the matrix is symmetrical and small (2x2 for line
    fit and 3x3 for circle fit), so the SelfAdjointEigenSolver from Eigen
    library is used, with the computedDirect  method (available only for 2x2
    and 3x3 Matrix) wich computes eigendecomposition of given matrix using a
    fast closed-form algorithm.
    For this optimization the matrix type must be known at compiling time.

*/

__host__ __device__ inline Vector3d min_eigen3D(const Matrix3d& A, double& chi2)
{
#if RFIT_DEBUG
    printf("min_eigen3D - enter\n");
#endif
    SelfAdjointEigenSolver<Matrix3d> solver(3);
    solver.computeDirect(A);
    int min_index;
    chi2 = solver.eigenvalues().minCoeff(&min_index);
#if RFIT_DEBUG
    printf("min_eigen3D - exit\n");
#endif
    return solver.eigenvectors().col(min_index);
}

/*!
    \brief A faster version of min_eigen3D() where double precision is not
    needed.

    \param A the Matrix you want to know eigenvector and eigenvalue.
    \param chi2 the double were the chi2-related quantity will be stored

    \return the eigenvector associated to the minimum eigenvalue.

    \detail The computedDirect() method of SelfAdjointEigenSolver for 3x3 Matrix
    indeed, use trigonometry function (it solves a third degree equation) which
    speed up in  single precision.
*/

__host__ __device__ inline Vector3d min_eigen3D_fast(const Matrix3d& A)
{
    SelfAdjointEigenSolver<Matrix3f> solver(3);
    solver.computeDirect(A.cast<float>());
    int min_index;
    solver.eigenvalues().minCoeff(&min_index);
    return solver.eigenvectors().col(min_index).cast<double>();
}

/*!
    \brief 2D version of min_eigen3D().

    \param A the Matrix you want to know eigenvector and eigenvalue.
    \param chi2 the double were the chi2-related quantity will be stored

    \return the eigenvector associated to the minimum eigenvalue.

    \detail The computedDirect() method of SelfAdjointEigenSolver for 2x2 Matrix
    do not use special math function (just sqrt) therefore it doesn't speed up
    significantly in single precision.
*/

__host__ __device__ inline Vector2d min_eigen2D(const Matrix2d& A, double& chi2)
{
    SelfAdjointEigenSolver<Matrix2d> solver(2);
    solver.computeDirect(A);
    int min_index;
    chi2 = solver.eigenvalues().minCoeff(&min_index);
    return solver.eigenvectors().col(min_index);
}

/*!
    \brief A very fast helix fit: it fits a circle by three points (first, middle
    and last point) and a line by two points (first and last).

    \param hits points to be fitted

    \return result in this form: (X0,Y0,R,tan(theta)).

    \warning points must be passed ordered (from internal layer to external) in
    order to maximize accuracy and do not mistake tan(theta) sign.

    \details This fast fit is used as pre-fit which is needed for:
    - weights estimation and chi2 computation in line fit (fundamental);
    - weights estimation and chi2 computation in circle fit (useful);
    - computation of error due to multiple scattering.
*/

__host__ __device__ inline Vector4d Fast_fit(const Matrix3xNd& hits)
{
    Vector4d result;
    u_int n = hits.cols();  // get the number of hits
    printIt(&hits, "Fast_fit - hits: ");

    // CIRCLE FIT
    // Make segments between middle-to-first(b) and last-to-first(c) hits
    const Vector2d b = hits.block(0, n / 2, 2, 1) - hits.block(0, 0, 2, 1);
    const Vector2d c = hits.block(0, n - 1, 2, 1) - hits.block(0, 0, 2, 1);
    printIt(&b, "Fast_fit - b: ");
    printIt(&c, "Fast_fit - c: ");
    // Compute their lengths
    const double b2 = b.squaredNorm();
    const double c2 = c.squaredNorm();
    double X0;
    double Y0;
    // The algebra has been verified (MR). The usual approach has been followed:
    // * use an orthogonal reference frame passing from the first point.
    // * build the segments (chords)
    // * build orthogonal lines through mid points
    // * make a system and solve for X0 and Y0.
    // * add the initial point
    if (abs(b.x()) > abs(b.y()))
    {  //!< in case b.x is 0 (2 hits with same x)
        const double k = c.x() / b.x();
        const double div = 2. * (k * b.y() - c.y());
        // if aligned TO FIX
        Y0 = (k * b2 - c2) / div;
        X0 = b2 / (2 * b.x()) - b.y() / b.x() * Y0;
    }
    else
    {
        const double k = c.y() / b.y();
        const double div = 2. * (k * b.x() - c.x());
        // if aligned TO FIX
        X0 = (k * b2 - c2) / div;
        Y0 = b2 / (2 * b.y()) - b.x() / b.y() * X0;
    }

    result(0) = X0 + hits(0, 0);
    result(1) = Y0 + hits(1, 0);
    result(2) = sqrt(sqr(X0) + sqr(Y0));
    printIt(&result, "Fast_fit - result: ");

    // LINE FIT
    const Vector2d d = hits.block(0, 0, 2, 1) - result.head(2);
    const Vector2d e = hits.block(0, n - 1, 2, 1) - result.head(2);
    printIt(&e, "Fast_fit - e: ");
    printIt(&d, "Fast_fit - d: ");
    // Compute the arc-length between first and last point: L = R * theta = R * atan (tan (Theta) )
    const double dr = result(2) * atan2(cross2D(d, e), d.dot(e));
    // Simple difference in Z between last and first hit
    const double dz = hits(2, n - 1) - hits(2, 0);

    result(3) = (dr / dz);

#if RFIT_DEBUG
    printf("Fast_fit: [%f, %f, %f, %f]\n", result(0), result(1), result(2), result(3));
#endif
    return result;
}

/*!
    \brief Fit a generic number of 2D points with a circle using Riemann-Chernov
    algorithm. Covariance matrix of fitted parameter is optionally computed.
    Multiple scattering (currently only in barrel layer) is optionally handled.

    \param hits2D 2D points to be fitted.
    \param hits_cov2D covariance matrix of 2D points.
    \param fast_fit pre-fit result in this form: (X0,Y0,R,tan(theta)).
    (tan(theta) is not used).
    \param B magnetic field
    \param error flag for error computation.
    \param scattering flag for multiple scattering

    \return circle circle_fit:
    -par parameter of the fitted circle in this form (X0,Y0,R); \n
    -cov covariance matrix of the fitted parameter (not initialized if
    error = false); \n
    -q charge of the particle; \n
    -chi2.

    \warning hits must be passed ordered from inner to outer layer (double hits
    on the same layer must be ordered too) so that multiple scattering is
    treated properly.
    \warning Multiple scattering for barrel is still not tested.
    \warning Multiple scattering for endcap hits is not handled (yet). Do not
    fit endcap hits with scattering = true !

    \bug for small pt (<0.3 Gev/c) chi2 could be slightly underestimated.
    \bug further investigation needed for error propagation with multiple
    scattering.
*/

__host__ __device__ inline circle_fit Circle_fit(const Matrix2xNd& hits2D,
                                                 const Matrix2Nd& hits_cov2D,
                                                 const Vector4d& fast_fit,
                                                 const VectorNd& rad,
                                                 const double B,
                                                 const bool error = true)
{
#if RFIT_DEBUG
    printf("circle_fit - enter\n");
#endif
    // INITIALIZATION
    Matrix2Nd V = hits_cov2D;
    u_int n = hits2D.cols();
    printIt(&hits2D, "circle_fit - hits2D:");
    printIt(&hits_cov2D, "circle_fit - hits_cov2D:");

#if RFIT_DEBUG
    printf("circle_fit - WEIGHT COMPUTATION\n");
#endif
    // WEIGHT COMPUTATION
    VectorNd weight;
    MatrixNd G;
    double renorm;
    {
        MatrixNd cov_rad;
        cov_rad = cov_carttorad_prefit(hits2D, V, fast_fit, rad);
        printIt(&cov_rad, "circle_fit - cov_rad:");
        // cov_rad = cov_carttorad(hits2D, V);

        MatrixNd scatter_cov_rad = Scatter_cov_rad(hits2D, fast_fit, rad, B);
        printIt(&scatter_cov_rad, "circle_fit - scatter_cov_rad:");
        printIt(&hits2D, "circle_fit - hits2D bis:");
#if RFIT_DEBUG
        printf("Address of hits2D: a) %p\n", &hits2D);
#endif
        V += cov_radtocart(hits2D, scatter_cov_rad, rad);
        printIt(&V, "circle_fit - V:");
        cov_rad += scatter_cov_rad;
        printIt(&cov_rad, "circle_fit - cov_rad:");
        Matrix4d cov_rad4 = cov_rad;
        Matrix4d G4;
        G4 = cov_rad4.inverse();
        printIt(&G4, "circle_fit - G4:");
        renorm = G4.sum();
        G4 *= 1. / renorm;
        printIt(&G4, "circle_fit - G4:");
        G = G4;
        weight = Weight_circle(G);
    }
    printIt(&weight, "circle_fit - weight:");

    // SPACE TRANSFORMATION
#if RFIT_DEBUG
    printf("circle_fit - SPACE TRANSFORMATION\n");
#endif

    // center
#if RFIT_DEBUG
    printf("Address of hits2D: b) %p\n", &hits2D);
#endif
    const Vector2d h_ = hits2D.rowwise().mean();  // centroid
    printIt(&h_, "circle_fit - h_:");
    Matrix3xNd p3D(3, n);
    p3D.block(0, 0, 2, n) = hits2D.colwise() - h_;
    printIt(&p3D, "circle_fit - p3D: a)");
    Vector2Nd mc(2 * n);  // centered hits, used in error computation
    mc << p3D.row(0).transpose(), p3D.row(1).transpose();
    printIt(&mc, "circle_fit - mc(centered hits):");

    // scale
    const double q = mc.squaredNorm();
    const double s = sqrt(n * 1. / q);  // scaling factor
    p3D *= s;

    // project on paraboloid
    p3D.row(2) = p3D.block(0, 0, 2, n).colwise().squaredNorm();
    printIt(&p3D, "circle_fit - p3D: b)");

#if RFIT_DEBUG
    printf("circle_fit - COST FUNCTION\n");
#endif
    // COST FUNCTION

    // compute
    Matrix3d A = Matrix3d::Zero();
    const Vector3d r0 = p3D * weight;  // center of gravity
    const Matrix3xNd X = p3D.colwise() - r0;
    A = X * G * X.transpose();
    printIt(&A, "circle_fit - A:");

#if RFIT_DEBUG
    printf("circle_fit - MINIMIZE\n");
#endif
    // minimize
    double chi2;
    Vector3d v = min_eigen3D(A, chi2);
#if RFIT_DEBUG
    printf("circle_fit - AFTER MIN_EIGEN\n");
#endif
    printIt(&v, "v BEFORE INVERSION");
    v *= (v(2) > 0) ? 1 : -1;  // TO FIX dovrebbe essere N(3)>0
    printIt(&v, "v AFTER INVERSION");
    // This hack to be able to run on GPU where the automatic assignment to a
    // double from the vector multiplication is not working.
#if RFIT_DEBUG
    printf("circle_fit - AFTER MIN_EIGEN 1\n");
#endif
    Matrix<double, 1, 1> cm;
#if RFIT_DEBUG
    printf("circle_fit - AFTER MIN_EIGEN 2\n");
#endif
    cm = -v.transpose() * r0;
#if RFIT_DEBUG
    printf("circle_fit - AFTER MIN_EIGEN 3\n");
#endif
    const double c = cm(0, 0);
    //  const double c = -v.transpose() * r0;

#if RFIT_DEBUG
    printf("circle_fit - COMPUTE CIRCLE PARAMETER\n");
#endif
    // COMPUTE CIRCLE PARAMETER

    // auxiliary quantities
    const double h = sqrt(1. - sqr(v(2)) - 4. * c * v(2));
    const double v2x2_inv = 1. / (2. * v(2));
    const double s_inv = 1. / s;
    Vector3d par_uvr_;  // used in error propagation
    par_uvr_ << -v(0) * v2x2_inv, -v(1) * v2x2_inv, h * v2x2_inv;

    circle_fit circle;
    circle.par << par_uvr_(0) * s_inv + h_(0), par_uvr_(1) * s_inv + h_(1), par_uvr_(2) * s_inv;
    circle.q = Charge(hits2D, circle.par);
    circle.chi2 = abs(chi2) * renorm * 1. / sqr(2 * v(2) * par_uvr_(2) * s);
    printIt(&circle.par, "circle_fit - CIRCLE PARAMETERS:");
    printIt(&circle.cov, "circle_fit - CIRCLE COVARIANCE:");
#if RFIT_DEBUG
    printf("circle_fit - CIRCLE CHARGE: %ld\n", circle.q);
#endif

#if RFIT_DEBUG
    printf("circle_fit - ERROR PROPAGATION\n");
#endif
    // ERROR PROPAGATION
    if (error)
    {
#if RFIT_DEBUG
        printf("circle_fit - ERROR PRPAGATION ACTIVATED\n");
#endif
        ArrayNd Vcs_[2][2];  // cov matrix of center & scaled points
#if RFIT_DEBUG
        printf("circle_fit - ERROR PRPAGATION ACTIVATED 2\n");
#endif
        {
            Matrix<double, 1, 1> cm;
            Matrix<double, 1, 1> cm2;
            cm = mc.transpose() * V * mc;
            //      cm2 = mc * mc.transpose();
            const double c = cm(0, 0);
            //      const double c2 = cm2(0,0);
            const Matrix2Nd Vcs = sqr(s) * V + sqr(sqr(s)) * 1. / (4. * q * n) *
                                                   (2. * V.squaredNorm() + 4. * c) *  // mc.transpose() * V * mc) *
                                                   mc * mc.transpose();
            printIt(&Vcs, "circle_fit - Vcs:");
            Vcs_[0][0] = Vcs.block(0, 0, n, n);
            Vcs_[0][1] = Vcs.block(0, n, n, n);
            Vcs_[1][1] = Vcs.block(n, n, n, n);
            Vcs_[1][0] = Vcs_[0][1].transpose();
            printIt(&Vcs, "circle_fit - Vcs:");
        }

        MatrixNd C[3][3];  // cov matrix of 3D transformed points
        {
            const ArrayNd t0 = (VectorXd::Constant(n, 1.) * p3D.row(0));
            const ArrayNd t1 = (VectorXd::Constant(n, 1.) * p3D.row(1));
            const ArrayNd t00 = p3D.row(0).transpose() * p3D.row(0);
            const ArrayNd t01 = p3D.row(0).transpose() * p3D.row(1);
            const ArrayNd t11 = p3D.row(1).transpose() * p3D.row(1);
            const ArrayNd t10 = t01.transpose();
            C[0][0] = Vcs_[0][0];
            C[0][1] = Vcs_[0][1];
            C[0][2] = 2. * (Vcs_[0][0] * t0 + Vcs_[0][1] * t1);
            C[1][1] = Vcs_[1][1];
            C[1][2] = 2. * (Vcs_[1][0] * t0 + Vcs_[1][1] * t1);
            C[2][2] = 2. * (Vcs_[0][0] * Vcs_[0][0] + Vcs_[0][0] * Vcs_[0][1] + Vcs_[1][1] * Vcs_[1][0] +
                            Vcs_[1][1] * Vcs_[1][1]) +
                      4. * (Vcs_[0][0] * t00 + Vcs_[0][1] * t01 + Vcs_[1][0] * t10 + Vcs_[1][1] * t11);
        }
        printIt(&C[0][0], "circle_fit - C[0][0]:");

        Matrix3d C0;  // cov matrix of center of gravity (r0.x,r0.y,r0.z)
        for (u_int i = 0; i < 3; ++i)
        {
            for (u_int j = i; j < 3; ++j)
            {
                Matrix<double, 1, 1> tmp;
                tmp = weight.transpose() * C[i][j] * weight;
                const double c = tmp(0, 0);
                C0(i, j) = c;  //weight.transpose() * C[i][j] * weight;
                C0(j, i) = C0(i, j);
            }
        }
        printIt(&C0, "circle_fit - C0:");

        const MatrixNd W = weight * weight.transpose();
        const MatrixNd H = MatrixXd::Identity(n, n).rowwise() - weight.transpose();
        const MatrixNx3d s_v = H * p3D.transpose();
        printIt(&W, "circle_fit - W:");
        printIt(&H, "circle_fit - H:");
        printIt(&s_v, "circle_fit - s_v:");

        MatrixNd D_[3][3];  // cov(s_v)
        {
            D_[0][0] = (H * C[0][0] * H.transpose()).cwiseProduct(W);
            D_[0][1] = (H * C[0][1] * H.transpose()).cwiseProduct(W);
            D_[0][2] = (H * C[0][2] * H.transpose()).cwiseProduct(W);
            D_[1][1] = (H * C[1][1] * H.transpose()).cwiseProduct(W);
            D_[1][2] = (H * C[1][2] * H.transpose()).cwiseProduct(W);
            D_[2][2] = (H * C[2][2] * H.transpose()).cwiseProduct(W);
            D_[1][0] = D_[0][1].transpose();
            D_[2][0] = D_[0][2].transpose();
            D_[2][1] = D_[1][2].transpose();
        }
        printIt(&D_[0][0], "circle_fit - D_[0][0]:");

        constexpr u_int nu[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        Matrix6d E;  // cov matrix of the 6 independent elements of A
        for (u_int a = 0; a < 6; ++a)
        {
            const u_int i = nu[a][0], j = nu[a][1];
            for (u_int b = a; b < 6; ++b)
            {
                const u_int k = nu[b][0], l = nu[b][1];
                VectorNd t0(n);
                VectorNd t1(n);
                if (l == k)
                {
                    t0 = 2. * D_[j][l] * s_v.col(l);
                    if (i == j)
                        t1 = t0;
                    else
                        t1 = 2. * D_[i][l] * s_v.col(l);
                }
                else
                {
                    t0 = D_[j][l] * s_v.col(k) + D_[j][k] * s_v.col(l);
                    if (i == j)
                        t1 = t0;
                    else
                        t1 = D_[i][l] * s_v.col(k) + D_[i][k] * s_v.col(l);
                }

                if (i == j)
                {
                    Matrix<double, 1, 1> cm;
                    cm = s_v.col(i).transpose() * (t0 + t1);
                    const double c = cm(0, 0);
                    E(a, b) = 0. + c;
                }
                else
                {
                    Matrix<double, 1, 1> cm;
                    cm = (s_v.col(i).transpose() * t0) + (s_v.col(j).transpose() * t1);
                    const double c = cm(0, 0);
                    E(a, b) = 0. + c;  //(s_v.col(i).transpose() * t0) + (s_v.col(j).transpose() * t1);
                }
                if (b != a)
                    E(b, a) = E(a, b);
            }
        }
        printIt(&E, "circle_fit - E:");

        Matrix<double, 3, 6> J2;  // Jacobian of min_eigen() (numerically computed)
        for (u_int a = 0; a < 6; ++a)
        {
            const u_int i = nu[a][0], j = nu[a][1];
            Matrix3d Delta = Matrix3d::Zero();
            Delta(i, j) = Delta(j, i) = abs(A(i, j) * d);
            J2.col(a) = min_eigen3D_fast(A + Delta);
            const int sign = (J2.col(a)(2) > 0) ? 1 : -1;
            J2.col(a) = (J2.col(a) * sign - v) / Delta(i, j);
        }
        printIt(&J2, "circle_fit - J2:");

        Matrix4d Cvc;  // joint cov matrix of (v0,v1,v2,c)
        {
            Matrix3d t0 = J2 * E * J2.transpose();
            Vector3d t1 = -t0 * r0;
            Cvc.block(0, 0, 3, 3) = t0;
            Cvc.block(0, 3, 3, 1) = t1;
            Cvc.block(3, 0, 1, 3) = t1.transpose();
            Matrix<double, 1, 1> cm1;
            //      Matrix<double, 1, 1> cm2;
            Matrix<double, 1, 1> cm3;
            cm1 = (v.transpose() * C0 * v);
            //      cm2 = (C0.cwiseProduct(t0)).sum();
            cm3 = (r0.transpose() * t0 * r0);
            const double c = cm1(0, 0) + (C0.cwiseProduct(t0)).sum() + cm3(0, 0);
            Cvc(3, 3) = c;
            // (v.transpose() * C0 * v) + (C0.cwiseProduct(t0)).sum() + (r0.transpose() * t0 * r0);
        }
        printIt(&Cvc, "circle_fit - Cvc:");

        Matrix<double, 3, 4> J3;  // Jacobian (v0,v1,v2,c)->(X0,Y0,R)
        {
            const double t = 1. / h;
            J3 << -v2x2_inv, 0, v(0) * sqr(v2x2_inv) * 2., 0, 0, -v2x2_inv, v(1) * sqr(v2x2_inv) * 2., 0,
                0, 0, -h * sqr(v2x2_inv) * 2. - (2. * c + v(2)) * v2x2_inv * t, -t;
        }
        printIt(&J3, "circle_fit - J3:");

        const RowVector2Nd Jq = mc.transpose() * s * 1. / n;  // var(q)
        printIt(&Jq, "circle_fit - Jq:");

        Matrix3d cov_uvr = J3 * Cvc * J3.transpose() * sqr(s_inv)  // cov(X0,Y0,R)
                           + (par_uvr_ * par_uvr_.transpose()) * (Jq * V * Jq.transpose());

        circle.cov = cov_uvr;
    }

    printIt(&circle.cov, "Circle cov:");
#if RFIT_DEBUG
    printf("circle_fit - exit\n");
#endif
    return circle;
}

/*!
    \brief Fit of helix parameter cotan(theta)) and Zip by projection on the
    pre-fitted cylinder  and line fit on its surface.

    \param hits hits coordinates.
    \param hits_cov covariance matrix of the hits.
    \param circle cylinder parameter, their covariance (if computed, otherwise
    uninitialized) and particle charge.
    \param fast_fit result of the previous fast fit in this form:
    (X0,Y0,R,cotan(theta))).
    \param error flag for error computation.

    \return line line_fit:
    -par parameter of the line in this form: (cotan(theta)), Zip); \n
    -cov covariance matrix of the fitted parameter; \n
    -chi2.

    \warning correlation between R and z are neglected, this could be relevant
    if geometry detector provides sloped modules in the R/z plane.

    \bug chi2 and errors could be slightly underestimated for small eta (<0.2)
    when pt is small (<0.3 Gev/c).

    \todo multiple scattering treatment.

    \details Line fit is made by orthogonal distance regression where
    correlation between coordinates in the transverse plane (x,y) and z are
    neglected (for a barrel + endcap geometry this is a very good
    approximation).
    Covariance matrix of the fitted parameter is optionally computed.
    Multiple scattering is not handled (yet).
    A fast pre-fit is performed in order to evaluate weights and to compute
    errors.
*/

__host__ __device__ inline line_fit Line_fit(const Matrix3xNd& hits,
                                             const Matrix3Nd& hits_cov,
                                             const circle_fit& circle,
                                             const Vector4d& fast_fit,
                                             const double B,
                                             const bool error = true)
{
    u_int n = hits.cols();
    // PROJECTION ON THE CILINDER
    Matrix2xNd p2D(2, n);
    MatrixNx5d Jx(n, 5);

#if RFIT_DEBUG
    printf("Line_fit - B: %g\n", B);

    printIt(&hits, "Line_fit points: ");
    printIt(&hits_cov, "Line_fit covs: ");
#endif
    // x & associated Jacobian
    // cfr https://indico.cern.ch/event/663159/contributions/2707659/attachments/1517175/2368189/Riemann_fit.pdf
    // Slide 11
    // a ==> -o i.e. the origin of the circle in XY plane, negative
    // b ==> p i.e. distances of the points wrt the origin of the circle.
    const Vector2d o(circle.par(0), circle.par(1));

    // associated Jacobian, used in weights and errors computation
    for (u_int i = 0; i < n; ++i)
    {  // x
        Vector2d p = hits.block(0, i, 2, 1) - o;
        const double cross = cross2D(-o, p);
        const double dot = (-o).dot(p);
        // atan2(cross, dot) give back the angle in the transverse plane so tha the
        // final equation reads: x_i = -q*R*theta (theta = angle returned by atan2)
        const double atan2_ = -circle.q * atan2(cross, dot);
        p2D(0, i) = atan2_ * circle.par(2);

        // associated Jacobian, used in weights and errors computation
        const double temp0 = -circle.q * circle.par(2) * 1. / (sqr(dot) + sqr(cross));
        double d_X0 = 0, d_Y0 = 0, d_R = 0.;  // good approximation for big pt and eta
        if (error)
        {
            d_X0 = -temp0 * ((p(1) + o(1)) * dot - (p(0) - o(0)) * cross);
            d_Y0 = temp0 * ((p(0) + o(0)) * dot - (o(1) - p(1)) * cross);
            d_R = atan2_;
        }
        const double d_x = temp0 * (o(1) * dot + o(0) * cross);
        const double d_y = temp0 * (-o(0) * dot + o(1) * cross);
        Jx.row(i) << d_X0, d_Y0, d_R, d_x, d_y;
    }
    // Math of d_{X0,Y0,R,x,y} all verified by hand

    // y
    p2D.row(1) = hits.row(2);

    // WEIGHT COMPUTATION
    Matrix2Nd cov_sz = MatrixXd::Zero(2 * n, 2 * n);
    VectorNd x_err2 = X_err2(hits_cov, circle, Jx, error, n);
    VectorNd y_err2 = hits_cov.block(2 * n, 2 * n, n, n).diagonal();
    cov_sz.block(0, 0, n, n) = x_err2.asDiagonal();
    cov_sz.block(n, n, n, n) = y_err2.asDiagonal();
#if RFIT_DEBUG
    printIt(&cov_sz, "line_fit - cov_sz:");
#endif
    MatrixNd cov_with_ms = Scatter_cov_line(cov_sz, fast_fit, p2D.row(0), p2D.row(1), B);
#if RFIT_DEBUG
    printIt(&cov_with_ms, "line_fit - cov_with_ms: ");
#endif
    Matrix4d G, G4;
    G4 = cov_with_ms.inverse();
#if RFIT_DEBUG
    printIt(&G4, "line_fit - cov_with_ms.inverse():");
#endif
    double renorm = G4.sum();
    G4 *= 1. / renorm;
#if RFIT_DEBUG
    printIt(&G4, "line_fit - G4:");
#endif
    G = G4;
    const VectorNd weight = Weight_circle(G);


    VectorNd err2_inv = cov_with_ms.diagonal();
    err2_inv = err2_inv.cwiseInverse();
//    const VectorNd err2_inv = Weight_line(x_err2, y_err2, fast_fit(3));
//    const VectorNd weight = err2_inv * 1. / err2_inv.sum();

#if RFIT_DEBUG
    printIt(&x_err2, "Line_fit - x_err2: ");
    printIt(&y_err2, "Line_fit - y_err2: ");
    printIt(&err2_inv, "Line_fit - err2_inv: ");
    printIt(&weight, "Line_fit - weight: ");
#endif

    // COST FUNCTION

    // compute
    // r0 represents the weighted mean of "x" and "y".
    const Vector2d r0 = p2D * weight;
    // This is the X  vector that will be used to build the
    // scatter matrix S = X^T * X
    const Matrix2xNd X = p2D.colwise() - r0;
    Matrix2d A = Matrix2d::Zero();
    A = X * G * X.transpose();
//    for (u_int i = 0; i < n; ++i)
//    {
//        A += err2_inv(i) * (X.col(i) * X.col(i).transpose());
//    }

#if RFIT_DEBUG
    printIt(&A, "Line_fit - A: ");
#endif

    // minimize
    double chi2;
    Vector2d v = min_eigen2D(A, chi2);
#if RFIT_DEBUG
    printIt(&v, "Line_fit - v: ");
    printf("Line_fit chi2: %e\n", chi2);
#endif

    // n *= (chi2>0) ? 1 : -1; //TO FIX
    // This hack to be able to run on GPU where the automatic assignment to a
    // double from the vector multiplication is not working.
    Matrix<double, 1, 1> cm;
    cm = -v.transpose() * r0;
    const double c = cm(0, 0);

    // COMPUTE LINE PARAMETER
    line_fit line;
    line.par << -v(0) / v(1),                          // cotan(theta))
        -c * sqrt(sqr(v(0)) + sqr(v(1))) * 1. / v(1);  // Zip
    line.chi2 = abs(chi2);
#if RFIT_DEBUG
    printIt(&(line.par), "Line_fit - line.par: ");
    printf("Line_fit - v norm: %e\n", sqrt(v(0)*v(0) + v(1)*v(1)));
#endif

    // ERROR PROPAGATION
    if (error)
    {
        const double v0_2 = sqr(v(0));
        const double v1_2 = sqr(v(1));

        Matrix3d C;  // cov(v,c)
        {
          // The norm is taken from Chernov, properly adapted to the weights case.
            double norm = v.transpose() * A * v;
            norm /= weight.sum();
#if RFIT_DEBUG
            printf("Line_fit - norm:    %e\n", norm);
#endif
            const double sig2 = 1. / (A(0, 0) + A(1, 1)) * norm;
            C(0, 0) = sig2 * v1_2;
            C(1, 1) = sig2 * v0_2;
            C(0, 1) = C(1, 0) = -sig2 * v(0) * v(1);
            const VectorNd weight_2 = (weight).array().square();
            const Vector2d C0(weight_2.dot(x_err2), weight_2.dot(y_err2));
            C.block(0, 2, 2, 1) = C.block(2, 0, 1, 2).transpose() = -C.block(0, 0, 2, 2) * r0;
            Matrix<double, 1, 1> tmp = (r0.transpose() * C.block(0, 0, 2, 2) * r0);
            C(2, 2) = v0_2 * C0(0) + v1_2 * C0(1) + C0(0) * C(0, 0) + C0(1) * C(1, 1) + tmp(0, 0);
        }
#if RFIT_DEBUG
        printIt(&C, "line_fit - C:");
#endif

        Matrix<double, 2, 3> J;  // Jacobian of (v,c) -> (cotan(theta)),Zip)
        {
            const double t0 = 1. / v(1);
            const double t1 = sqr(t0);
            const double sqrt_ = sqrt(v1_2 + v0_2);
            const double t2 = 1. / sqrt_;
            J << -t0, v(0) * t1, 0, -c * v(0) * t0 * t2, v0_2 * c * t1 * t2, -sqrt_ * t0;
        }
        Matrix<double, 3, 2> JT = J.transpose().eval();
#if RFIT_DEBUG
        printIt(&J, "line_fit - J:");
#endif
        line.cov = J * C * JT;
    }

#if RFIT_DEBUG
    printIt(&line.cov, "Line cov:");
#endif
    return line;
}

/*!
    \brief Helix fit by three step:
    -fast pre-fit (see Fast_fit() for further info); \n
    -circle fit of hits projected in the transverse plane by Riemann-Chernov
        algorithm (see Circle_fit() for further info); \n
    -line fit of hits projected on cylinder surface by orthogonal distance
        regression (see Line_fit for further info). \n
    Points must be passed ordered (from inner to outer layer).

    \param hits Matrix3xNd hits coordinates in this form: \n
        |x0|x1|x2|...|xn| \n
        |y0|y1|y2|...|yn| \n
        |z0|z1|z2|...|zn|

    \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n

   |(x0,x0)|(x1,x0)|(x2,x0)|.|(y0,x0)|(y1,x0)|(y2,x0)|.|(z0,x0)|(z1,x0)|(z2,x0)| \n
   |(x0,x1)|(x1,x1)|(x2,x1)|.|(y0,x1)|(y1,x1)|(y2,x1)|.|(z0,x1)|(z1,x1)|(z2,x1)| \n
   |(x0,x2)|(x1,x2)|(x2,x2)|.|(y0,x2)|(y1,x2)|(y2,x2)|.|(z0,x2)|(z1,x2)|(z2,x2)| \n
       .       .       .    .    .       .       .    .    .       .       .     \n
   |(x0,y0)|(x1,y0)|(x2,y0)|.|(y0,y0)|(y1,y0)|(y2,x0)|.|(z0,y0)|(z1,y0)|(z2,y0)| \n
   |(x0,y1)|(x1,y1)|(x2,y1)|.|(y0,y1)|(y1,y1)|(y2,x1)|.|(z0,y1)|(z1,y1)|(z2,y1)| \n
   |(x0,y2)|(x1,y2)|(x2,y2)|.|(y0,y2)|(y1,y2)|(y2,x2)|.|(z0,y2)|(z1,y2)|(z2,y2)| \n
       .       .       .    .    .       .       .    .    .       .       .     \n
   |(x0,z0)|(x1,z0)|(x2,z0)|.|(y0,z0)|(y1,z0)|(y2,z0)|.|(z0,z0)|(z1,z0)|(z2,z0)| \n
   |(x0,z1)|(x1,z1)|(x2,z1)|.|(y0,z1)|(y1,z1)|(y2,z1)|.|(z0,z1)|(z1,z1)|(z2,z1)| \n
   |(x0,z2)|(x1,z2)|(x2,z2)|.|(y0,z2)|(y1,z2)|(y2,z2)|.|(z0,z2)|(z1,z2)|(z2,z2)|

   \param B magnetic field in the center of the detector in Gev/cm/c
   unit, in order to perform pt calculation.
   \param error flag for error computation.
   \param scattering flag for multiple scattering treatment.
   (see Circle_fit() documentation for further info).

   \warning see Circle_fit(), Line_fit() and Fast_fit() warnings.

   \bug see Circle_fit(), Line_fit() and Fast_fit() bugs.
*/

inline helix_fit Helix_fit(const Matrix3xNd& hits, const Matrix3Nd& hits_cov, const double B,
                           const bool error = true)
{
    u_int n = hits.cols();
    VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

    // Fast_fit gives back (X0, Y0, R, theta) w/o errors, using only 3 points.
    const Vector4d fast_fit = Fast_fit(hits);

    circle_fit circle = Circle_fit(hits.block(0, 0, 2, n),
                                   hits_cov.block(0, 0, 2 * n, 2 * n),
                                   fast_fit, rad, B, error);
    line_fit line = Line_fit(hits, hits_cov, circle, fast_fit, B, error);

    par_uvrtopak(circle, B, error);

    helix_fit helix;
    helix.par << circle.par, line.par;
    if (error)
    {
        helix.cov = MatrixXd::Zero(5, 5);
        helix.cov.block(0, 0, 3, 3) = circle.cov;
        helix.cov.block(3, 3, 2, 2) = line.cov;
    }
    helix.q = circle.q;
    helix.chi2_circle = circle.chi2;
    helix.chi2_line = line.chi2;

    return helix;
}

}  // namespace Rfit

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h
