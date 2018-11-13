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

/*!  Compute the Radiation length in the uniform hypothesis
 *
 * The Pixel detector, barrel and forward, is considered as an omogeneous
 * cilinder of material, whose radiation lengths has been derived from the TDR
 * plot that shows that 16cm correspond to 0.06 radiation lengths. Therefore
 * one radiation length corresponds to 16cm/0.06 =~ 267 cm. All radiation
 * lengths are computed using this unique number, in both regions, barrel and
 * endcap.
 *
 * NB: no angle corrections nor projections are computed inside this routine.
 * It is therefore the responsibility of the caller to supply the proper
 * lengths in input. These lenghts are the path travelled by the particle along
 * its trajectory, namely the so called S of the helix in 3D space.
 *
 * \param length_values vector of incremental distances that will be translated
 * into radiation length equivalent. Each radiation length i is computed
 * incrementally with respect to the previous length i-1. The first lenght has
 * no reference point (i.e. it has the dca).
 *
 * \return incremental radiation lengths that correspond to each segment.
 */

__host__ __device__ inline
void computeRadLenUniformMaterial(const VectorNd &length_values,
    VectorNd & rad_lengths) {
  // Radiation length of the pixel detector in the uniform assumption, with
  // 0.06 rad_len at 16 cm
  const double XX_0 = 16.f/(0.06);
//  const double XX_0 = 1000.*16.f/(0.06);
  u_int n = length_values.rows();
  rad_lengths(0) = length_values(0)/XX_0;
  for (u_int j = 1; j < n; ++j) {
    rad_lengths(j) = std::abs(length_values(j)-length_values(j-1))/XX_0;
  }
}

/*!
    \brief Compute the covariance matrix along cartesian S-Z of points due to
    multiple Coulomb scattering to be used in the line_fit, for the barrel
    and forward cases.

    The input covariance matrix is in the variables s-z, original and
    unrotated.

    The multiple scattering component is computed in the usual linear
    approximation, using the 3D path which is computed as the squared root of
    the squared sum of the s and z components passed in.

    Internally a rotation by theta is performed and the covariance matrix
    returned is the one in the direction orthogonal to the rotated S3D axis,
    i.e. along the rotated Z axis.

    The choice of the rotation is not arbitrary, but derived from the fact that
    putting the horizontal axis along the S3D direction allows the usage of the
    ordinary least squared fitting techiques with the trivial parametrization y
    = mx + q, avoiding the patological case with m = +/- inf, that would
    correspond to the case at eta = 0.
 */

__host__ __device__ inline MatrixNd Scatter_cov_line(Matrix2Nd& cov_sz,
                                                     const Vector4d& fast_fit,
                                                     VectorNd const& s_arcs,
                                                     VectorNd const& z_values,
                                                     const double theta,
                                                     const double B)
{
#if RFIT_DEBUG
    Rfit::printIt(&s_arcs, "Scatter_cov_line - s_arcs: ");
#endif
    u_int n = s_arcs.rows();
    double p_t = fast_fit(2) * B;
    double p_2 = p_t * p_t * (1. + 1. / (fast_fit(3) * fast_fit(3)));
    VectorNd rad_lengths_S(n);
    // See documentation at http://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
    // Basically, to perform cwise operations on Matrices and Vectors, you need
    // to transform them into Array-like objects.
    VectorNd S_values = s_arcs.array() * s_arcs.array() + z_values.array() * z_values.array();
    S_values = S_values.array().sqrt();
    computeRadLenUniformMaterial(S_values, rad_lengths_S);
    VectorNd sig2_S(n);
    sig2_S = .000225 / p_2 * (1.f + 0.038 * rad_lengths_S.array().log()).abs2() * rad_lengths_S.array();
#if RFIT_DEBUG
    Rfit::printIt(&cov_sz, "Scatter_cov_line - cov_sz: ");
#endif
    Matrix2Nd rot = MatrixXd::Zero(2 * n, 2 * n);
    for (u_int i = 0; i < n; ++i) {
      rot(i, i) = sin(theta);
      rot(n + i, n + i) = sin(theta);
      u_int j = (i + n);
      rot(i, j) = i < j ? cos(theta) : -cos(theta);
    }

#if RFIT_DEBUG
    Rfit::printIt(&rot, "Scatter_cov_line - rot: ");
#endif

    Matrix2Nd tmp = rot*cov_sz*rot.transpose();
    for (u_int k = 0; k < n; ++k)
    {
      for (u_int l = k; l < n; ++l)
      {
        for (u_int i = 0; i < std::min(k, l); ++i)
        {
          tmp(k + n, l + n) += std::abs(S_values(k) - S_values(i)) * std::abs(S_values(l) - S_values(i)) * sig2_S(i);
          tmp(l + n, k + n) = tmp(k + n, l + n);
        }
      }
    }
    // We are interested only in the errors orthogonal to the rotated s-axis
    // which, in our formalism, are in the lower square matrix.
#if RFIT_DEBUG
    Rfit::printIt(&tmp, "Scatter_cov_line - tmp: ");
#endif
    return tmp.block(n, n, n, n);
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

    \details Only the tangential component is computed (the radial one is
    negligible).

 */
__host__ __device__ inline MatrixNd Scatter_cov_rad(const Matrix2xNd& p2D,
                                                    const Vector4d& fast_fit,
                                                    VectorNd const& rad,
                                                    double B)
{
    u_int n = p2D.cols();
    double p_t = fast_fit(2) * B;
    double p_2 = p_t * p_t * (1. + 1. / (fast_fit(3) * fast_fit(3)));
    double theta = atan(fast_fit(3));
    theta = theta < 0. ? theta + M_PI :  theta;
    VectorNd s_values(n);
    VectorNd rad_lengths(n);
    const Vector2d o(fast_fit(0), fast_fit(1));

    // associated Jacobian, used in weights and errors computation
    for (u_int i = 0; i < n; ++i)
    {  // x
        Vector2d p = p2D.block(0, i, 2, 1) - o;
        const double cross = cross2D(-o, p);
        const double dot = (-o).dot(p);
        const double atan2_ = atan2(cross, dot);
        s_values(i) = std::abs(atan2_ * fast_fit(2));
    }
    computeRadLenUniformMaterial(s_values*sqrt(1. + 1./(fast_fit(3)*fast_fit(3))), rad_lengths);
    MatrixNd scatter_cov_rad = MatrixXd::Zero(n, n);
    VectorNd sig2(n);
    sig2 = .000225 / p_2 * (1.f + 0.038 * rad_lengths.array().log()).abs2() * rad_lengths.array();
    for (u_int k = 0; k < n; ++k)
    {
        for (u_int l = k; l < n; ++l)
        {
            for (u_int i = 0; i < std::min(k, l); ++i)
            {
              scatter_cov_rad(k, l) += (rad(k) - rad(i)) * (rad(l) - rad(i)) * sig2(i) / (sqr(sin(theta)));
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
                v(0)*v2x2_inv*t, v(1)*v2x2_inv*t, -h * sqr(v2x2_inv) * 2. - (2. * c + v(2)) * v2x2_inv * t, -t;
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

__host__ __device__ inline line_fit Line_fit_odr(const Matrix3xNd& hits,
    const Matrix3Nd& hits_cov,
    const circle_fit& circle,
    const Vector4d& fast_fit,
    const double B,
    const bool error = true)
{
    u_int n = hits.cols();
    double theta = -circle.q*atan(fast_fit(3));
    theta = theta < 0. ? theta + M_PI :  theta;
    // PROJECTION ON THE CILINDER
    Matrix2xNd p2D = MatrixXd::Zero(2, n);
    Eigen::Matrix<double, 2, 6> Jx;

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
    Matrix2Nd cov_sz = MatrixXd::Zero(2 * n, 2 * n);
    for (u_int i = 0; i < n; ++i)
    {  // x
      Matrix6d Cov = MatrixXd::Zero(6, 6);
      Matrix2d Cov_sz_single = MatrixXd::Zero(2, 2);
        Vector2d p = hits.block(0, i, 2, 1) - o;
        const double cross = cross2D(-o, p);
        const double dot = (-o).dot(p);
        // atan2(cross, dot) give back the angle in the transverse plane so tha the
        // final equation reads: x_i = -q*R*theta (theta = angle returned by atan2)
        const double atan2_ = -circle.q * atan2(cross, dot);
        p2D(0, i) = atan2_ * circle.par(2);

        // associated Jacobian, used in weights and errors- computation
        const double temp0 = -circle.q * circle.par(2) * 1. / (sqr(dot) + sqr(cross));
        double d_X0 = 0., d_Y0 = 0., d_R = 0.;  // good approximation for big pt and eta
        if (error)
        {
            d_X0 = -temp0 * ((p(1) + o(1)) * dot - (p(0) - o(0)) * cross);
            d_Y0 = temp0 * ((p(0) + o(0)) * dot - (o(1) - p(1)) * cross);
            d_R = atan2_;
        }
        const double d_x = temp0 * (o(1) * dot + o(0) * cross);
        const double d_y = temp0 * (-o(0) * dot + o(1) * cross);
        Jx << d_X0, d_Y0, d_R, d_x, d_y, 0., 0., 0., 0., 0., 0., 1.;
//        Jx << d_X0, d_Y0, d_R, p(1)/p.norm(), -p(0)/p.norm(), 0, 0, 0, 0, 0, 0, 1.;
        Cov.block(0, 0, 3, 3) = circle.cov;
        Cov(3, 3) = hits_cov(i, i);
        Cov(4, 4) = hits_cov(i + n, i + n);
        Cov(5, 5) = hits_cov(i + 2*n, i + 2*n);
        Cov(3, 4) = Cov(4, 3) = hits_cov(i, i + n);
        Cov(3, 5) = Cov(5, 3) = hits_cov(i, i + 2*n);
        Cov(4, 5) = Cov(5, 4) = hits_cov(i + n, i + 2*n);
        Cov_sz_single = Jx * Cov * Jx.transpose();
        cov_sz(i, i) = Cov_sz_single(0, 0);
        cov_sz(i + n, i + n) = Cov_sz_single(1, 1);
        cov_sz(i, i + n) = cov_sz(i + n, i) = Cov_sz_single(0, 1);
    }
    // Math of d_{X0,Y0,R,x,y} all verified by hand

    // y
    p2D.row(1) = hits.row(2);

    // WEIGHT COMPUTATION
#if RFIT_DEBUG
    printIt(&cov_sz, "line_fit - cov_sz:");
#endif
    MatrixNd cov_with_ms = Scatter_cov_line(cov_sz, fast_fit, p2D.row(0), p2D.row(1), theta, B);
#if RFIT_DEBUG
    printIt(&cov_with_ms, "line_fit - cov_with_ms: ");
#endif
    Matrix4d G;
    G = cov_with_ms.inverse();
#if RFIT_DEBUG
    printIt(&G, "line_fit - cov_with_ms.inverse():");
#endif
    double renorm = G.sum();
    G *= 1. / renorm;
#if RFIT_DEBUG
    printIt(&G, "line_fit - G4:");
#endif

    const VectorNd weight = Weight_circle(G);

    // COST FUNCTION

    // compute
    // r0 represents the weighted mean of "x" and "y".
    const Vector2d r0 = p2D * weight;
    // This is the X  vector that will be used to build the
    // scatter matrix S = X^T * X
    const Matrix2xNd X = p2D.colwise() - r0;
    Matrix2d A = Matrix2d::Zero();
    A = X * G * X.transpose();

#if RFIT_DEBUG
    printIt(&A, "Line_fit - A: ");
#endif

    // minimize. v is normalized!!
    double chi2;
    Vector2d v = min_eigen2D(A, chi2);
#if RFIT_DEBUG
    printIt(&v, "Line_fit - v: ");
    printf("Line_fit chi2: %e\n", chi2);
#endif

    // This hack to be able to run on GPU where the automatic assignment to a
    // double from the vector multiplication is not working.
    Matrix<double, 1, 1> cm;
    cm = -v.transpose() * r0;
    const double c = cm(0, 0);

    // COMPUTE LINE PARAMETER
    line_fit line;
    line.par << -v(0) / v(1),                          // cotan(theta))
        -c / v(1);  // Zip
    line.chi2 = abs(chi2*renorm);
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
//            double norm_empirical = cov_with_ms.diagonal().mean();
#if RFIT_DEBUG
            printf("Chi_2: %g\n", chi2);
            printf("Norm: %g\n", norm);
            printf("weight.sum(): %g\n", weight.sum());
            printf("Line_fit - norm:    %e\n", norm);
#endif

            const double sig2 = norm/(A(0,0) + A(1,1));
            C(0, 0) = sig2 * v1_2;
            C(1, 1) = sig2 * v0_2;
            C(1, 0) = C(0, 1) = -sig2 * v(0) * v(1);
            C(2, 2) = sig2 * (v(0)*r0(1)-v(1)*r0(0))*(v(0)*r0(1)-v(1)*r0(0)) + (sig2/n)*(A(0,0)+A(1,1));
            C(0, 2) = C(2, 0) = sig2*(v(0)*r0(1)-v(1)*r0(0))*v(1);
            C(1, 2) = C(2, 1) = - sig2*(v(0)*r0(1)-v(1)*r0(0))*v(0);
        }
#if RFIT_DEBUG
        printIt(&C, "line_fit - C:");
#endif

        Matrix<double, 2, 3> J;  // Jacobian of (v,c) -> (cotan(theta)),Zip)
        {
            const double t0 = 1. / v(1);
            const double t1 = sqr(t0);
            J << -t0, v(0) * t1, 0., 0., c * t1, -t0;
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

/*!  \brief Perform an ordinary least square fit in the s-z plane to compute
 * the parameters cotTheta and Zip.
 *
 * The fit is performed in the rotated S3D-Z' plane, following the formalism of
 * Frodesen, Chapter 10, p. 259.
 *
 * The system has been rotated to both try to use the combined errors in s-z
 * along Z', as errors in the Y direction and to avoid the patological case of
 * degenerate lines with angular coefficient m = +/- inf.
 *
 * The rotation is using the information on the theta angle computed in the
 * fast fit. The rotation is such that the S3D axis will be the X-direction,
 * while the rotated Z-axis will be the Y-direction. This pretty much follows
 * what is done in the same fit in the Broken Line approach.
 */

__host__ __device__ inline line_fit Line_fit(const Matrix3xNd& hits,
    const Matrix3Nd& hits_cov,
    const circle_fit& circle,
    const Vector4d& fast_fit,
    const double B,
    const bool error = true) {
  auto n = hits.cols();
  double theta = -circle.q*atan(fast_fit(3));
  theta = theta < 0. ? theta + M_PI : theta;

  // PROJECTION ON THE CILINDER
  //
  // p2D will be:
  // [s1, s2, s3, ..., sn]
  // [z1, z2, z3, ..., zn]
  // s values will be ordinary x-values
  // z values will be ordinary y-values

  Matrix2xNd p2D(2, n);
  Eigen::Matrix<double, 2, 6> Jx;

  p2D << MatrixXd::Zero(2, n);
  Jx << MatrixXd::Zero(2, 6);

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
  Matrix2Nd cov_sz = MatrixXd::Zero(2 * n, 2 * n);
  Matrix6d Cov(6,6);
  Matrix2d Cov_sz_single(2, 2);
  for (u_int i = 0; i < n; ++i)
  {
    Vector2d p = hits.block(0, i, 2, 1) - o;
    const double cross = cross2D(-o, p);
    const double dot = (-o).dot(p);
    // atan2(cross, dot) give back the angle in the transverse plane so tha the
    // final equation reads: x_i = -q*R*theta (theta = angle returned by atan2)
    const double atan2_ = -circle.q * atan2(cross, dot);
//    p2D.coeffRef(1, i) = atan2_ * circle.par(2);
    p2D(0, i) = atan2_ * circle.par(2);

    // associated Jacobian, used in weights and errors- computation
    const double temp0 = -circle.q * circle.par(2) * 1. / (sqr(dot) + sqr(cross));
    double d_X0 = 0., d_Y0 = 0., d_R = 0.;  // good approximation for big pt and eta
    if (error)
    {
      d_X0 = -temp0 * ((p(1) + o(1)) * dot - (p(0) - o(0)) * cross);
      d_Y0 = temp0 * ((p(0) + o(0)) * dot - (o(1) - p(1)) * cross);
      d_R = atan2_;
    }
    const double d_x = temp0 * (o(1) * dot + o(0) * cross);
    const double d_y = temp0 * (-o(0) * dot + o(1) * cross);
    Jx << d_X0, d_Y0, d_R, d_x, d_y, 0., 0., 0., 0., 0., 0., 1.;

    Cov << MatrixXd::Zero(6, 6);
    Cov_sz_single << MatrixXd::Zero(2, 2);
    Cov.block(0, 0, 3, 3) = circle.cov;
    Cov(3, 3) = hits_cov(i, i);                        // x errors
    Cov(4, 4) = hits_cov(i + n, i + n);                // y errors
    Cov(5, 5) = hits_cov(i + 2*n, i + 2*n);            // z errors
    Cov(3, 4) = Cov(4, 3) = hits_cov(i, i + n);        // cov_xy
    Cov(3, 5) = Cov(5, 3) = hits_cov(i, i + 2*n);      // cov_xz
    Cov(4, 5) = Cov(5, 4) = hits_cov(i + n, i + 2*n);  // cov_yz
    Cov_sz_single = Jx * Cov * Jx.transpose();
    cov_sz(i, i) = Cov_sz_single(0, 0);
    cov_sz(i + n, i + n) = Cov_sz_single(1, 1);
    cov_sz(i, i + n) = cov_sz(i + n, i) = Cov_sz_single(0, 1);
  }
  // Math of d_{X0,Y0,R,x,y} all verified by hand
  p2D.row(1) = hits.row(2);

  // The following matrix will contain errors orthogonal to the rotated S
  // component only, with the Multiple Scattering properly treated!!
  MatrixNd cov_with_ms = Scatter_cov_line(cov_sz, fast_fit, p2D.row(0), p2D.row(1), theta, B);
#if RFIT_DEBUG
  printIt(&cov_sz, "line_fit - cov_sz:");
  printIt(&cov_with_ms, "line_fit - cov_with_ms: ");
#endif

  // Prepare the Rotation Matrix to rotate the points
  Eigen::Matrix<double, 2, 2> rot = Eigen::Matrix<double, 2, 2>::Zero();
  rot << sin(theta), cos(theta), -cos(theta), sin(theta);

  // Rotate Points with the shape [2, n]
  Matrix2xNd p2D_rot = rot*p2D;

#if RFIT_DEBUG
  printf("Fast fit Tan(theta): %g\n", fast_fit(3));
  printf("Rotation angle: %g\n", theta);
  printIt(&rot, "Rotation Matrix:");
  printIt(&p2D, "Original Hits(s,z):");
  printIt(&p2D_rot, "Rotated hits(S3D, Z'):");
  printIt(&rot, "Rotation Matrix:");
#endif

  // Build the A Matrix
  Matrix2xNd A(2,n);
  A << MatrixXd::Ones(1, n), p2D_rot.row(0);  // rotated s values

#if RFIT_DEBUG
  printIt(&A, "A Matrix:");
#endif

  // Build A^T V-1 A, where V-1 is the covariance of only the Y components.
  MatrixNd Vy_inv = cov_with_ms.inverse();
  Eigen::Matrix<double, 2, 2> Inv_Cov = A*Vy_inv*A.transpose();

  // Compute the Covariance Matrix of the fit parameters
  Eigen::Matrix<double, 2, 2> Cov_params = Inv_Cov.inverse();

  // Now Compute the Parameters in the form [2,1]
  // The first component is q.
  // The second component is m.
  Eigen::Matrix<double, 2, 1> sol = Cov_params*A*Vy_inv*p2D_rot.row(1).transpose();


#if RFIT_DEBUG
  printIt(&sol, "Rotated solutions:");
#endif

  // We need now to transfer back the results in the original s-z plane
  auto common_factor = 1./(sin(theta)-sol(1,0)*cos(theta));
  Matrix<double, 2, 2> J = Matrix<double, 2, 2>::Zero();
  J << 0., common_factor*common_factor, common_factor, sol(0,0)*cos(theta)*common_factor*common_factor;

  double m = common_factor*(sol(1,0)*sin(theta)+cos(theta));
  double q = common_factor*sol(0,0);
  auto cov_mq = J * Cov_params * J.transpose();

  VectorNd res = p2D_rot.row(1).transpose() - A.transpose() * sol;
  double chi2 = res.transpose()*Vy_inv*res;
  chi2 = chi2 / float(n);

  line_fit line;
  line.par << m, q;
  line.cov << cov_mq;
  line.chi2 = chi2;

#if RFIT_DEBUG
  printf("Common_factor: %g\n", common_factor);
  printIt(&J, "Jacobian:");
  printIt(&sol, "Rotated solutions:");
  printIt(&Cov_params, "Cov_params:");
  printIt(&cov_mq, "Rotated Covariance Matrix:");
  printIt(&(line.par), "Real Parameters:");
  printIt(&(line.cov), "Real Covariance Matrix:");
  printf("Chi2: %g\n", chi2);
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
