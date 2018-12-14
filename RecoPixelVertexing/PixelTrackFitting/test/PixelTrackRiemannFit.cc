#define _USE_MATH_DEFINES

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <memory>  // unique_ptr

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

using namespace std;
using namespace Eigen;
using namespace Rfit;
using std::unique_ptr;

namespace Rfit {
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Vector4i = Eigen::Matrix<int, 4, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector8d = Eigen::Matrix<double, 8, 1>;
};  // namespace Rfit

struct hits_gen {
  Matrix3xNd hits;
  Matrix3Nd hits_cov;
  Vector5d true_par;
};

struct geometry {
  Vector8d barrel;
  Vector4i barrel_2;
  Vector8d R_err;
  Vector8d Rp_err;
  Vector8d z_err;
  Vector6d hand;
  Vector3i hand_2;
  Vector6d xy_err;
  Vector6d zh_err;
  double z_max;
  double r_max;
};

void test_helix_fit();

constexpr int c_speed = 299792458;
constexpr double pi = M_PI;
default_random_engine generator(1);

void smearing(const Vector5d& err, const bool& isbarrel, double& x, double& y, double& z) {
  normal_distribution<double> dist_R(0., err[0]);
  normal_distribution<double> dist_Rp(0., err[1]);
  normal_distribution<double> dist_z(0., err[2]);
  normal_distribution<double> dist_xyh(0., err[3]);
  normal_distribution<double> dist_zh(0., err[4]);
  if (isbarrel) {
    double dev_Rp = dist_Rp(generator);
    double dev_R = dist_R(generator);
    double R = sqrt(Rfit::sqr(x) + Rfit::sqr(y));
    x += dev_Rp * +y / R + dev_R * -x / R;
    y += dev_Rp * -x / R + dev_R * -y / R;
    z += dist_z(generator);
  } else {
    x += dist_xyh(generator);
    y += dist_xyh(generator);
    z += dist_zh(generator);
  }
}

void Hits_cov(Matrix3Nd& V, const unsigned int& i, const unsigned int& n, const Matrix3xNd& hits,
              const Vector5d& err, bool isbarrel) {
  if (isbarrel) {
    double R2 = Rfit::sqr(hits(0, i)) + Rfit::sqr(hits(1, i));
    V(i, i) =
        (Rfit::sqr(err[1]) * Rfit::sqr(hits(1, i)) + Rfit::sqr(err[0]) * Rfit::sqr(hits(0, i))) /
        R2;
    V(i + n, i + n) =
        (Rfit::sqr(err[1]) * Rfit::sqr(hits(0, i)) + Rfit::sqr(err[0]) * Rfit::sqr(hits(1, i))) /
        R2;
    V(i, i + n) = V(i + n, i) =
        (Rfit::sqr(err[0]) - Rfit::sqr(err[1])) * hits(1, i) * hits(0, i) / R2;
    V(i + 2 * n, i + 2 * n) = Rfit::sqr(err[2]);
  } else {
    V(i, i) = Rfit::sqr(err[3]);
    V(i + n, i + n) = Rfit::sqr(err[3]);
    V(i + 2 * n, i + 2 * n) = Rfit::sqr(err[4]);
  }
}

hits_gen Hits_gen(const unsigned int& n, const Matrix<double, 6, 1>& gen_par) {
  hits_gen gen;
  gen.hits = MatrixXd::Zero(3, n);
  gen.hits_cov = MatrixXd::Zero(3 * n, 3 * n);
  // err /= 10000.;
  constexpr double rad[8] = {2.95, 6.8, 10.9, 16., 3.1, 7., 11., 16.2};
  // constexpr double R_err[8] = {5./10000, 5./10000, 5./10000, 5./10000, 5./10000,
  // 5./10000, 5./10000, 5./10000};  constexpr double Rp_err[8] = {35./10000, 18./10000,
  // 15./10000, 34./10000, 35./10000, 18./10000, 15./10000, 34./10000};  constexpr double z_err[8] =
  // {72./10000, 38./10000, 25./10000, 56./10000, 72./10000, 38./10000, 25./10000, 56./10000};
  constexpr double R_err[8] = {10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000,
                               10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000};
  constexpr double Rp_err[8] = {35. / 10000, 18. / 10000, 15. / 10000, 34. / 10000,
                                35. / 10000, 18. / 10000, 15. / 10000, 34. / 10000};
  constexpr double z_err[8] = {72. / 10000, 38. / 10000, 25. / 10000, 56. / 10000,
                               72. / 10000, 38. / 10000, 25. / 10000, 56. / 10000};
  const double x2 = gen_par(0) + gen_par(4) * cos(gen_par(3) * pi / 180);
  const double y2 = gen_par(1) + gen_par(4) * sin(gen_par(3) * pi / 180);
  const double alpha = atan2(y2, x2);

  for (unsigned int i = 0; i < n; ++i) {
    const double a = gen_par(4);
    const double b = rad[i];
    const double c = sqrt(Rfit::sqr(x2) + Rfit::sqr(y2));
    const double beta = acos((Rfit::sqr(a) - Rfit::sqr(b) - Rfit::sqr(c)) / (-2. * b * c));
    const double gamma = alpha + beta;
    gen.hits(0, i) = rad[i] * cos(gamma);
    gen.hits(1, i) = rad[i] * sin(gamma);
    gen.hits(2, i) = gen_par(2) + 1 / tan(gen_par(5) * pi / 180) * 2. *
                                      asin(sqrt(Rfit::sqr((gen_par(0) - gen.hits(0, i))) +
                                                Rfit::sqr((gen_par(1) - gen.hits(1, i)))) /
                                           (2. * gen_par(4))) *
                                      gen_par(4);
    // isbarrel(i) = ??
    Vector5d err;
    err << R_err[i], Rp_err[i], z_err[i], 0, 0;
    smearing(err, true, gen.hits(0, i), gen.hits(1, i), gen.hits(2, i));
    Hits_cov(gen.hits_cov, i, n, gen.hits, err, true);
  }

  return gen;
}

Vector5d True_par(const Matrix<double, 6, 1>& gen_par, const int& charge, const double& B_field) {
  Vector5d true_par;
  const double x0 = gen_par(0) + gen_par(4) * cos(gen_par(3) * pi / 180);
  const double y0 = gen_par(1) + gen_par(4) * sin(gen_par(3) * pi / 180);
  circle_fit circle;
  circle.par << x0, y0, gen_par(4);
  circle.q = 1;
  Rfit::par_uvrtopak(circle, B_field, false);
  true_par.block(0, 0, 3, 1) = circle.par;
  true_par(3) = 1 / tan(gen_par(5) * pi / 180);
  const int dir = ((gen_par(0) - cos(true_par(0) - pi / 2) * true_par(1)) * (gen_par(1) - y0) -
                       (gen_par(1) - sin(true_par(0) - pi / 2) * true_par(1)) * (gen_par(0) - x0) >
                   0)
                      ? -1
                      : 1;
  true_par(4) = gen_par(2) +
                1 / tan(gen_par(5) * pi / 180) * dir * 2.f *
                    asin(sqrt(Rfit::sqr((gen_par(0) - cos(true_par(0) - pi / 2) * true_par(1))) +
                              Rfit::sqr((gen_par(1) - sin(true_par(0) - pi / 2) * true_par(1)))) /
                         (2.f * gen_par(4))) *
                    gen_par(4);
  return true_par;
}

Matrix<double, 6, 1> New_par(const Matrix<double, 6, 1>& gen_par, const int& charge,
                             const double& B_field) {
  Matrix<double, 6, 1> new_par;
  new_par.block(0, 0, 3, 1) = gen_par.block(0, 0, 3, 1);
  new_par(3) = gen_par(3) - charge * 90;
  new_par(4) = gen_par(4) / B_field;
//  new_par(5) = atan(sinh(gen_par(5))) * 180 / pi;
  new_par(5) = 2.*atan(exp(-gen_par(5))) * 180 / pi;
  return new_par;
}

void test_helix_fit() {
  int n_;
  int iteration;
  int debug2 = 0;
  bool return_err;
  const double B_field = 3.8 * c_speed / pow(10, 9) / 100;
  Matrix<double, 6, 1> gen_par;
  Vector5d true_par;
  Vector5d err;
//  while (1) {
    generator.seed(1);
    int debug = 0;
    debug2 = 0;
    std::cout << std::setprecision(6);
    cout << "_________________________________________________________________________\n";
    cout << "n x(cm) y(cm) z(cm) phi(grad) R(Gev/c) eta iteration return_err debug" << endl;
//    cin >> n_ >> gen_par(0) >> gen_par(1) >> gen_par(2) >> gen_par(3) >> gen_par(4) >> gen_par(5) >>
//        iteration >> return_err >> debug2;
    n_ = 4;
    gen_par(0) = -0.1;  // x
    gen_par(1) = 0.1;   // y
    gen_par(2) = -1.;  // z
    gen_par(3) = 45.;   // phi
    gen_par(4) = 10.;   // R (p_t)
    gen_par(5) = 1.;   // eta
    iteration = 1;
    return_err = true;
    debug2 = 1;

    iteration *= 10;
    gen_par = New_par(gen_par, 1, B_field);
    true_par = True_par(gen_par, 1, B_field);
    Matrix3xNd hits;
    Matrix3Nd hits_cov;
    unique_ptr<helix_fit[]> helix(new helix_fit[iteration]);
//    helix_fit* helix = new helix_fit[iteration];
    Matrix<double, 41, Dynamic, 1> score(41, iteration);

    for (int i = 0; i < iteration; i++) {
      if (debug2 == 1 && i == (iteration - 1)) {
        debug = 1;
      }
      hits_gen gen;
      gen = Hits_gen(n_, gen_par);
//      gen.hits = MatrixXd::Zero(3, 4);
//      gen.hits_cov = MatrixXd::Zero(3 * 4, 3 * 4);
//      gen.hits.col(0) << 1.82917642593, 2.0411875248, 7.18495464325;
//      gen.hits.col(1) << 4.47041416168, 4.82704305649, 18.6394691467;
//      gen.hits.col(2) << 7.25991010666, 7.74653434753, 30.6931324005;
//      gen.hits.col(3) << 8.99161434174, 9.54262828827, 38.1338043213;
      helix[i] = Rfit::Helix_fit(gen.hits, gen.hits_cov, B_field, return_err);

      if (debug)
        cout << std::setprecision(10)
            << "phi:  " << helix[i].par(0) << " +/- " << sqrt(helix[i].cov(0, 0)) << " vs "
            << true_par(0) << endl
            << "Tip:  " << helix[i].par(1) << " +/- " << sqrt(helix[i].cov(1, 1)) << " vs "
            << true_par(1) << endl
            << "p_t:  " << helix[i].par(2) << " +/- " << sqrt(helix[i].cov(2, 2)) << " vs "
            << true_par(2) << endl
            << "theta:" << helix[i].par(3) << " +/- " << sqrt(helix[i].cov(3, 3)) << " vs "
            << true_par(3) << endl
            << "Zip:  " << helix[i].par(4) << " +/- " << sqrt(helix[i].cov(4, 4)) << " vs "
            << true_par(4) << endl
            << "charge:" << helix[i].q << " vs 1" << endl
            << "covariance matrix:" << endl
            << helix[i].cov << endl
            << "Initial hits:\n" << gen.hits << endl
            << "Initial Covariance:\n" << gen.hits_cov << endl;
    }

    for (int x = 0; x < iteration; x++) {
      // Compute PULLS information
      score(0, x) = (helix[x].par(0) - true_par(0)) / sqrt(helix[x].cov(0, 0));
      score(1, x) = (helix[x].par(1) - true_par(1)) / sqrt(helix[x].cov(1, 1));
      score(2, x) = (helix[x].par(2) - true_par(2)) / sqrt(helix[x].cov(2, 2));
      score(3, x) = (helix[x].par(3) - true_par(3)) / sqrt(helix[x].cov(3, 3));
      score(4, x) = (helix[x].par(4) - true_par(4)) / sqrt(helix[x].cov(4, 4));
      score(5, x) =
          (helix[x].par(0) - true_par(0)) * (helix[x].par(1) - true_par(1)) / (helix[x].cov(0, 1));
      score(6, x) =
          (helix[x].par(0) - true_par(0)) * (helix[x].par(2) - true_par(2)) / (helix[x].cov(0, 2));
      score(7, x) =
          (helix[x].par(1) - true_par(1)) * (helix[x].par(2) - true_par(2)) / (helix[x].cov(1, 2));
      score(8, x) =
          (helix[x].par(3) - true_par(3)) * (helix[x].par(4) - true_par(4)) / (helix[x].cov(3, 4));
      score(9, x) = helix[x].chi2_circle;
      score(25, x) = helix[x].chi2_line;
      score(10, x) = sqrt(helix[x].cov(0, 0)) / helix[x].par(0) * 100;
      score(13, x) = sqrt(helix[x].cov(3, 3)) / helix[x].par(3) * 100;
      score(14, x) = sqrt(helix[x].cov(4, 4)) / helix[x].par(4) * 100;
      score(15, x) = (helix[x].par(0) - true_par(0)) * (helix[x].par(3) - true_par(3)) /
                     sqrt(helix[x].cov(0, 0)) / sqrt(helix[x].cov(3, 3));
      score(16, x) = (helix[x].par(1) - true_par(1)) * (helix[x].par(3) - true_par(3)) /
                     sqrt(helix[x].cov(1, 1)) / sqrt(helix[x].cov(3, 3));
      score(17, x) = (helix[x].par(2) - true_par(2)) * (helix[x].par(3) - true_par(3)) /
                     sqrt(helix[x].cov(2, 2)) / sqrt(helix[x].cov(3, 3));
      score(18, x) = (helix[x].par(0) - true_par(0)) * (helix[x].par(4) - true_par(4)) /
                     sqrt(helix[x].cov(0, 0)) / sqrt(helix[x].cov(4, 4));
      score(19, x) = (helix[x].par(1) - true_par(1)) * (helix[x].par(4) - true_par(4)) /
                     sqrt(helix[x].cov(1, 1)) / sqrt(helix[x].cov(4, 4));
      score(20, x) = (helix[x].par(2) - true_par(2)) * (helix[x].par(4) - true_par(4)) /
                     sqrt(helix[x].cov(2, 2)) / sqrt(helix[x].cov(4, 4));
      score(21, x) = (helix[x].par(0) - true_par(0)) * (helix[x].par(1) - true_par(1)) /
                     sqrt(helix[x].cov(0, 0)) / sqrt(helix[x].cov(1, 1));
      score(22, x) = (helix[x].par(0) - true_par(0)) * (helix[x].par(2) - true_par(2)) /
                     sqrt(helix[x].cov(0, 0)) / sqrt(helix[x].cov(2, 2));
      score(23, x) = (helix[x].par(1) - true_par(1)) * (helix[x].par(2) - true_par(2)) /
                     sqrt(helix[x].cov(1, 1)) / sqrt(helix[x].cov(2, 2));
      score(24, x) = (helix[x].par(3) - true_par(3)) * (helix[x].par(4) - true_par(4)) /
                     sqrt(helix[x].cov(3, 3)) / sqrt(helix[x].cov(4, 4));
    }

    double phi_ = score.row(0).mean();
    double a_ = score.row(1).mean();
    double pt_ = score.row(2).mean();
    double coT_ = score.row(3).mean();
    double Zip_ = score.row(4).mean();
    Matrix5d correlation;
    correlation << 1., score.row(21).mean(), score.row(22).mean(), score.row(15).mean(),
        score.row(20).mean(), score.row(21).mean(), 1., score.row(23).mean(), score.row(16).mean(),
        score.row(19).mean(), score.row(22).mean(), score.row(23).mean(), 1., score.row(17).mean(),
        score.row(20).mean(), score.row(15).mean(), score.row(16).mean(), score.row(17).mean(), 1.,
        score.row(24).mean(), score.row(18).mean(), score.row(19).mean(), score.row(20).mean(),
        score.row(24).mean(), 1.;

    cout << "\nPULLS:\n"
         << "phi:  " << phi_ << "     "
         << sqrt((score.row(0).array() - phi_).square().sum() / (iteration - 1)) << "   "
         << abs(score.row(10).mean()) << "%\n"
         << "a0 :  " << a_ << "     "
         << sqrt((score.row(1).array() - a_).square().sum() / (iteration - 1)) << "   "
         << abs(score.row(11).mean()) << "%\n"
         << "pt :  " << pt_ << "     "
         << sqrt((score.row(2).array() - pt_).square().sum() / (iteration - 1)) << "   "
         << abs(score.row(12).mean()) << "%\n"
         << "coT:  " << coT_ << "     "
         << sqrt((score.row(3).array() - coT_).square().sum() / (iteration - 1)) << "   "
         << abs(score.row(13).mean()) << "%\n"
         << "Zip:  " << Zip_ << "     "
         << sqrt((score.row(4).array() - Zip_).square().sum() / (iteration - 1)) << "   "
         << abs(score.row(14).mean()) << "%\n\n"
         << "cov(phi,a0)_:  " << score.row(5).mean() << "\n"
         << "cov(phi,pt)_:  " << score.row(6).mean() << "\n"
         << "cov(a0,pt)_:   " << score.row(7).mean() << "\n"
         << "cov(coT,Zip)_: " << score.row(8).mean() << "\n\n"
         << "chi2_circle:  " << score.row(9).mean() << " vs " << n_ - 3 << "\n"
         << "chi2_line:    " << score.row(25).mean() << " vs " << n_ - 2 << "\n\n"
         << "correlation matrix:\n"
         << correlation << "\n\n"
         << endl;
//  }
}

int main() {
  test_helix_fit();
  return 0;
}
