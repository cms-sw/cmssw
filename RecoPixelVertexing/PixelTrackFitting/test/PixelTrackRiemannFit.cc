#define _USE_MATH_DEFINES

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <memory>  // unique_ptr
#include<chrono>

#include <TFile.h>
#include <TH1F.h>

#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
//#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"

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

// quadruplets...
struct hits_gen {
  Matrix3xNd<4> hits;
  Eigen::Matrix<float,6,4> hits_ge;
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

template<int N>
void Hits_cov(Eigen::Matrix<float,6,4> & V, const unsigned int& i, const unsigned int& n, const Matrix3xNd<N>& hits,
              const Vector5d& err, bool isbarrel) {
  if (isbarrel) {
    double R2 = Rfit::sqr(hits(0, i)) + Rfit::sqr(hits(1, i));
    V.col(i)[0] =
        (Rfit::sqr(err[1]) * Rfit::sqr(hits(1, i)) + Rfit::sqr(err[0]) * Rfit::sqr(hits(0, i))) /
        R2;
    V.col(i)[2] =
        (Rfit::sqr(err[1]) * Rfit::sqr(hits(0, i)) + Rfit::sqr(err[0]) * Rfit::sqr(hits(1, i))) /
        R2;
    V.col(i)[1] =
        (Rfit::sqr(err[0]) - Rfit::sqr(err[1])) * hits(1, i) * hits(0, i) / R2;
    V.col(i)[5] = Rfit::sqr(err[2]);
  } else {
    V.col(i)[0] = Rfit::sqr(err[3]);
    V.col(i)[2] = Rfit::sqr(err[3]);
    V.col(i)[5] = Rfit::sqr(err[4]);
  }
}

hits_gen Hits_gen(const unsigned int& n, const Matrix<double, 6, 1>& gen_par) {
  hits_gen gen;
  gen.hits = MatrixXd::Zero(3, n);
  gen.hits_ge = Eigen::Matrix<float,6,4>::Zero();
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
    Hits_cov(gen.hits_ge, i, n, gen.hits, err, true);
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

template<typename Fit, size_t N>
void computePull(std::array<Fit, N> & fit, const char * label,
    int n_, int iteration, const Vector5d & true_par) {
  Eigen::Matrix<double, 41, Eigen::Dynamic, 1> score(41, iteration);

  std::string histo_name("Phi Pull");
  histo_name += label;
  TH1F phi_pull(histo_name.data(), histo_name.data(), 100, -10., 10.);
  histo_name = "dxy Pull ";
  histo_name += label;
  TH1F dxy_pull(histo_name.data(), histo_name.data(), 100, -10., 10.);
  histo_name = "dz Pull ";
  histo_name += label;
  TH1F dz_pull(histo_name.data(), histo_name.data(), 100, -10., 10.);
  histo_name = "Theta Pull ";
  histo_name += label;
  TH1F theta_pull(histo_name.data(), histo_name.data(), 100, -10., 10.);
  histo_name = "Pt Pull ";
  histo_name += label;
  TH1F pt_pull(histo_name.data(), histo_name.data(), 100, -10., 10.);
  histo_name = "Phi Error ";
  histo_name += label;
  TH1F phi_error(histo_name.data(), histo_name.data(), 100, 0., 0.1);
  histo_name = "dxy error ";
  histo_name += label;
  TH1F dxy_error(histo_name.data(), histo_name.data(), 100, 0., 0.1);
  histo_name = "dz error ";
  histo_name += label;
  TH1F dz_error(histo_name.data(), histo_name.data(), 100, 0., 0.1);
  histo_name = "Theta error ";
  histo_name += label;
  TH1F theta_error(histo_name.data(), histo_name.data(), 100, 0., 0.1);
  histo_name = "Pt error ";
  histo_name += label;
  TH1F pt_error(histo_name.data(), histo_name.data(), 100, 0., 0.1);
  for (int x = 0; x < iteration; x++) {
    // Compute PULLS information
    score(0, x) = (fit[x].par(0) - true_par(0)) / sqrt(fit[x].cov(0, 0));
    score(1, x) = (fit[x].par(1) - true_par(1)) / sqrt(fit[x].cov(1, 1));
    score(2, x) = (fit[x].par(2) - true_par(2)) / sqrt(fit[x].cov(2, 2));
    score(3, x) = (fit[x].par(3) - true_par(3)) / sqrt(fit[x].cov(3, 3));
    score(4, x) = (fit[x].par(4) - true_par(4)) / sqrt(fit[x].cov(4, 4));
    phi_pull.Fill(score(0, x));
    dxy_pull.Fill(score(1, x));
    pt_pull.Fill(score(2, x));
    theta_pull.Fill(score(3, x));
    dz_pull.Fill(score(4, x));
    phi_error.Fill(sqrt(fit[x].cov(0, 0)));
    dxy_error.Fill(sqrt(fit[x].cov(1, 1)));
    pt_error.Fill(sqrt(fit[x].cov(2, 2)));
    theta_error.Fill(sqrt(fit[x].cov(3, 3)));
    dz_error.Fill(sqrt(fit[x].cov(4, 4)));
    score(5, x) =
      (fit[x].par(0) - true_par(0)) * (fit[x].par(1) - true_par(1)) / (fit[x].cov(0, 1));
    score(6, x) =
      (fit[x].par(0) - true_par(0)) * (fit[x].par(2) - true_par(2)) / (fit[x].cov(0, 2));
    score(7, x) =
      (fit[x].par(1) - true_par(1)) * (fit[x].par(2) - true_par(2)) / (fit[x].cov(1, 2));
    score(8, x) =
      (fit[x].par(3) - true_par(3)) * (fit[x].par(4) - true_par(4)) / (fit[x].cov(3, 4));
    score(9, x) = fit[x].chi2_circle;
    score(25, x) = fit[x].chi2_line;
    score(10, x) = sqrt(fit[x].cov(0, 0)) / fit[x].par(0) * 100;
    score(13, x) = sqrt(fit[x].cov(3, 3)) / fit[x].par(3) * 100;
    score(14, x) = sqrt(fit[x].cov(4, 4)) / fit[x].par(4) * 100;
    score(15, x) = (fit[x].par(0) - true_par(0)) * (fit[x].par(3) - true_par(3)) /
      sqrt(fit[x].cov(0, 0)) / sqrt(fit[x].cov(3, 3));
    score(16, x) = (fit[x].par(1) - true_par(1)) * (fit[x].par(3) - true_par(3)) /
      sqrt(fit[x].cov(1, 1)) / sqrt(fit[x].cov(3, 3));
    score(17, x) = (fit[x].par(2) - true_par(2)) * (fit[x].par(3) - true_par(3)) /
      sqrt(fit[x].cov(2, 2)) / sqrt(fit[x].cov(3, 3));
    score(18, x) = (fit[x].par(0) - true_par(0)) * (fit[x].par(4) - true_par(4)) /
      sqrt(fit[x].cov(0, 0)) / sqrt(fit[x].cov(4, 4));
    score(19, x) = (fit[x].par(1) - true_par(1)) * (fit[x].par(4) - true_par(4)) /
      sqrt(fit[x].cov(1, 1)) / sqrt(fit[x].cov(4, 4));
    score(20, x) = (fit[x].par(2) - true_par(2)) * (fit[x].par(4) - true_par(4)) /
      sqrt(fit[x].cov(2, 2)) / sqrt(fit[x].cov(4, 4));
    score(21, x) = (fit[x].par(0) - true_par(0)) * (fit[x].par(1) - true_par(1)) /
      sqrt(fit[x].cov(0, 0)) / sqrt(fit[x].cov(1, 1));
    score(22, x) = (fit[x].par(0) - true_par(0)) * (fit[x].par(2) - true_par(2)) /
      sqrt(fit[x].cov(0, 0)) / sqrt(fit[x].cov(2, 2));
    score(23, x) = (fit[x].par(1) - true_par(1)) * (fit[x].par(2) - true_par(2)) /
      sqrt(fit[x].cov(1, 1)) / sqrt(fit[x].cov(2, 2));
    score(24, x) = (fit[x].par(3) - true_par(3)) * (fit[x].par(4) - true_par(4)) /
      sqrt(fit[x].cov(3, 3)) / sqrt(fit[x].cov(4, 4));
    score(30, x) = fit[x].par(0);
    score(31, x) = fit[x].par(1);
    score(32, x) = fit[x].par(2);
    score(33, x) = fit[x].par(3);
    score(34, x) = fit[x].par(4);
    score(35, x) = sqrt(fit[x].cov(0,0));
    score(36, x) = sqrt(fit[x].cov(1,1));
    score(37, x) = sqrt(fit[x].cov(2,2));
    score(38, x) = sqrt(fit[x].cov(3,3));
    score(39, x) = sqrt(fit[x].cov(4,4));

  }

  double phi_ = score.row(0).mean();
  double a_ = score.row(1).mean();
  double pt_ = score.row(2).mean();
  double coT_ = score.row(3).mean();
  double Zip_ = score.row(4).mean();
  std::cout << std::setprecision(5) << std::scientific << label << " AVERAGE FITTED VALUES: \n"
    << "phi: " << score.row(30).mean() << " +/- " << score.row(35).mean() << " [+/-] " << sqrt(score.row(35).array().abs2().mean() - score.row(35).mean()*score.row(35).mean()) << std::endl
    << "d0:  " << score.row(31).mean() << " +/- " << score.row(36).mean() << " [+/-] " << sqrt(score.row(36).array().abs2().mean() - score.row(36).mean()*score.row(36).mean()) << std::endl
    << "pt:  " << score.row(32).mean() << " +/- " << score.row(37).mean() << " [+/-] " << sqrt(score.row(37).array().abs2().mean() - score.row(37).mean()*score.row(37).mean()) << std::endl
    << "coT: " << score.row(33).mean() << " +/- " << score.row(38).mean() << " [+/-] " << sqrt(score.row(38).array().abs2().mean() - score.row(38).mean()*score.row(38).mean()) << std::endl
    << "Zip: " << score.row(34).mean() << " +/- " << score.row(39).mean() << " [+/-] " << sqrt(score.row(39).array().abs2().mean() - score.row(39).mean()*score.row(39).mean()) << std::endl;

  Matrix5d correlation;
  correlation << 1., score.row(21).mean(), score.row(22).mean(), score.row(15).mean(),
              score.row(20).mean(), score.row(21).mean(), 1., score.row(23).mean(), score.row(16).mean(),
              score.row(19).mean(), score.row(22).mean(), score.row(23).mean(), 1., score.row(17).mean(),
              score.row(20).mean(), score.row(15).mean(), score.row(16).mean(), score.row(17).mean(), 1.,
              score.row(24).mean(), score.row(18).mean(), score.row(19).mean(), score.row(20).mean(),
              score.row(24).mean(), 1.;

  cout << "\n" << label << " PULLS (mean, sigma, relative_error):\n"
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

  phi_pull.Fit("gaus", "Q");
  dxy_pull.Fit("gaus", "Q");
  dz_pull.Fit("gaus", "Q");
  theta_pull.Fit("gaus", "Q");
  pt_pull.Fit("gaus", "Q");
  phi_pull.Write();
  dxy_pull.Write();
  dz_pull.Write();
  theta_pull.Write();
  pt_pull.Write();
  phi_error.Write();
  dxy_error.Write();
  dz_error.Write();
  theta_error.Write();
  pt_error.Write();
}


void test_helix_fit(bool getcin) {
  int n_;
  bool return_err;
  const double B_field = 3.8 * c_speed / pow(10, 9) / 100;
  Matrix<double, 6, 1> gen_par;
  Vector5d true_par;
  Vector5d err;
  generator.seed(1);
  std::cout << std::setprecision(6);
  cout << "_________________________________________________________________________\n";
  cout << "n x(cm) y(cm) z(cm) phi(grad) R(Gev/c) eta iteration return_err debug" << endl;
  if (getcin) {
    cout << "hits: ";
    cin  >> n_;
    cout << "x: ";
    cin  >> gen_par(0);
    cout << "y: ";
    cin  >> gen_par(1);
    cout << "z: ";
    cin  >> gen_par(2);
    cout << "phi: ";
    cin  >> gen_par(3);
    cout << "p_t: ";
    cin  >> gen_par(4);
    cout << "eta: ";
    cin  >> gen_par(5);
  } else {
     n_ = 4;
     gen_par(0) = -0.1;  // x
     gen_par(1) = 0.1;   // y
     gen_par(2) = -1.;  // z
     gen_par(3) = 45.;   // phi
     gen_par(4) = 10.;   // R (p_t)
     gen_par(5) = 1.;   // eta
  }
  return_err = true;

  const int iteration = 5000;
  gen_par = New_par(gen_par, 1, B_field);
  true_par = True_par(gen_par, 1, B_field);
  // Matrix3xNd<4> hits;
  std::array<helix_fit, iteration> helixRiemann_fit;
//  std::array<BrokenLine::helix_fit, iteration> helixBrokenLine_fit;

  std::cout << "\nTrue parameters: "
    << "phi: " << true_par(0) << " "
    << "dxy: " << true_par(1) << " "
    << "pt: " << true_par(2) << " "
    << "CotT: " << true_par(3) << " "
    << "Zip: " << true_par(4) << " "
    << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start-start;
  for (int i = 0; i < 100*iteration; i++) {
    hits_gen gen;
    gen = Hits_gen(n_, gen_par);
    //      gen.hits = MatrixXd::Zero(3, 4);
    //      gen.hits_cov = MatrixXd::Zero(3 * 4, 3 * 4);
    //      gen.hits.col(0) << 1.82917642593, 2.0411875248, 7.18495464325;
    //      gen.hits.col(1) << 4.47041416168, 4.82704305649, 18.6394691467;
    //      gen.hits.col(2) << 7.25991010666, 7.74653434753, 30.6931324005;
    //      gen.hits.col(3) << 8.99161434174, 9.54262828827, 38.1338043213;
    delta -= std::chrono::high_resolution_clock::now()-start;
    helixRiemann_fit[i%iteration] = Rfit::Helix_fit(gen.hits, gen.hits_ge, B_field, return_err);
    delta += std::chrono::high_resolution_clock::now()-start;

//    helixBrokenLine_fit[i] = BrokenLine::Helix_fit(gen.hits, gen.hits_cov, B_field);

    if (helixRiemann_fit[i%iteration].par(0)>10.) std::cout << "error" << std::endl;
    if (0==i)
      cout << std::setprecision(6)
        << "phi:  " << helixRiemann_fit[i].par(0) << " +/- " << sqrt(helixRiemann_fit[i].cov(0, 0)) << " vs "
        << true_par(0) << endl
        << "Tip:  " << helixRiemann_fit[i].par(1) << " +/- " << sqrt(helixRiemann_fit[i].cov(1, 1)) << " vs "
        << true_par(1) << endl
        << "p_t:  " << helixRiemann_fit[i].par(2) << " +/- " << sqrt(helixRiemann_fit[i].cov(2, 2)) << " vs "
        << true_par(2) << endl
        << "theta:" << helixRiemann_fit[i].par(3) << " +/- " << sqrt(helixRiemann_fit[i].cov(3, 3)) << " vs "
        << true_par(3) << endl
        << "Zip:  " << helixRiemann_fit[i].par(4) << " +/- " << sqrt(helixRiemann_fit[i].cov(4, 4)) << " vs "
        << true_par(4) << endl
        << "charge:" << helixRiemann_fit[i].q << " vs 1" << endl
        << "covariance matrix:" << endl
        << helixRiemann_fit[i].cov << endl
        << "Initial hits:\n" << gen.hits << endl
        << "Initial Covariance:\n" << gen.hits_ge << endl;
        
  }
  std::cout << "elapsted time " << double(std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count())/1.e6 << std::endl;
  computePull(helixRiemann_fit, "Riemann", n_, iteration, true_par);
//  computePull(helixBrokenLine_fit, "BrokenLine", n_, iteration, true_par);
}

int main(int nargs, char**) {
  TFile f("TestFitResults.root", "RECREATE");
  test_helix_fit(nargs>1);
  f.Close();
  return 0;
}

