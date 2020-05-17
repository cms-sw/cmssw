#include "PhysicsTools/Utilities/interface/Integral.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <sys/time.h>
#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TROOT.h"
#include "TH2.h"

using namespace funct;
using namespace std;

double getTime() {
  struct timeval t;
  if (gettimeofday(&t, nullptr) < 0)
    abort();
  return (double)t.tv_sec + (double(t.tv_usec) * 1E-6);
}

struct gauss {
  static const double c;
  double operator()(double x) const { return c * exp(-x * x); }
};

const double gauss::c = 2. / sqrt(M_PI);

struct gauss1 : public gauss {};

struct gauss2 : public gauss {};

struct gauss3 : public gauss {};

struct gauss4 : public gauss {};

struct gaussPrimitive {
  double operator()(double x) const { return erf(x); }
};

NUMERICAL_FUNCT_INTEGRAL(gauss1, TrapezoidIntegrator);

NUMERICAL_FUNCT_INTEGRAL(gauss2, GaussLegendreIntegrator);

NUMERICAL_FUNCT_INTEGRAL(gauss3, GaussIntegrator);

NUMERICAL_FUNCT_INTEGRAL(gauss4, RootIntegrator);

template <typename G, typename I>
pair<double, double> check(const G &g, const I &i) {
  gaussPrimitive pr;
  double xMax = 1;
  double i0 = pr(xMax);
  double t0, t1;
  t0 = getTime();
  double i1 = integral_f(g, 0, xMax, i);
  t1 = getTime();
  pair<double, double> p = make_pair(t1 - t0, fabs(i1 - i0));
  cout << ">>> time: " << p.first << ", accuracy: " << p.second << endl;
  return p;
}

int main() {
  gauss1 g1;
  gauss2 g2;
  gauss3 g3;
  gauss4 g4;

  cout << ">>> Trapezoidal integration" << endl;
  vector<pair<double, double> > t;
  for (size_t i = 2; i < 20; i += 1) {
    TrapezoidIntegrator i1(i);
    t.push_back(check(g1, i1));
  }
  for (size_t i = 20; i < 1000; i += 20) {
    TrapezoidIntegrator i1(i);
    t.push_back(check(g1, i1));
  }
  cout << ">>> Gauss Legendre integration" << endl;
  vector<pair<double, double> > l;
  for (size_t i = 1; i < 20; i += 1) {
    GaussLegendreIntegrator i2(i, 1.e-5);
    l.push_back(check(g2, i2));
  }
  for (size_t i = 20; i < 1000; i += 20) {
    GaussLegendreIntegrator i2(i, 1.e-5);
    l.push_back(check(g2, i2));
  }
  cout << ">>> Gauss integration" << endl;
  vector<pair<double, double> > g;
  for (double e = 100; e > 1.e-5; e /= 2) {
    GaussIntegrator i3(e);
    g.push_back(check(g3, i3));
  }
  cout << ">>> ROOT GSL integration" << endl;
  vector<pair<double, double> > r;
  for (double e = 100; e > 1.e-5; e /= 2) {
    RootIntegrator i4(ROOT::Math::IntegrationOneDim::kADAPTIVESINGULAR, e, e);
    r.push_back(check(g4, i4));
  }
  gROOT->SetStyle("Plain");
  TCanvas canvas;
  canvas.SetLogx();
  canvas.SetLogy();
  double xMin = 1e6, xMax = 0, yMin = 1e6, yMax = 0;
  size_t nt = t.size();
  double *xt = new double[nt], *yt = new double[nt];
  const double xAbsMin = 1e-10, yAbsMin = 1e-10;
  for (size_t i = 0; i < nt; ++i) {
    xt[i] = t[i].first;
    yt[i] = t[i].second;
    if (xt[i] < xAbsMin)
      xt[i] = xAbsMin;
    if (yt[i] < yAbsMin)
      yt[i] = yAbsMin;
    if (xMin > xt[i])
      xMin = xt[i];
    if (xMax < xt[i])
      xMax = xt[i];
    if (yMin > yt[i])
      yMin = yt[i];
    if (yMax < yt[i])
      yMax = yt[i];
  }
  size_t nl = l.size();
  double *xl = new double[nl], *yl = new double[nl];
  for (size_t i = 0; i < nl; ++i) {
    xl[i] = l[i].first;
    yl[i] = l[i].second;
    if (xl[i] < xAbsMin)
      xl[i] = xAbsMin;
    if (yl[i] < yAbsMin)
      yl[i] = yAbsMin;
    if (xMin > xl[i])
      xMin = xl[i];
    if (xMax < xl[i])
      xMax = xl[i];
    if (yMin > yl[i])
      yMin = yl[i];
    if (yMax < yl[i])
      yMax = yl[i];
  }
  size_t ng = g.size();
  double *xg = new double[ng], *yg = new double[ng];
  for (size_t i = 0; i < ng; ++i) {
    xg[i] = g[i].first;
    yg[i] = g[i].second;
    if (xg[i] < xAbsMin)
      xg[i] = xAbsMin;
    if (yg[i] < yAbsMin)
      yg[i] = yAbsMin;
    if (xMin > xg[i])
      xMin = xg[i];
    if (xMax < xg[i])
      xMax = xg[i];
    if (yMin > yg[i])
      yMin = yg[i];
    if (yMax < yg[i])
      yMax = yg[i];
  }
  size_t nr = r.size();
  double *xr = new double[nr], *yr = new double[nr];
  for (size_t i = 0; i < nr; ++i) {
    xr[i] = r[i].first;
    yr[i] = r[i].second;
    if (xr[i] < xAbsMin)
      xr[i] = xAbsMin;
    if (yr[i] < yAbsMin)
      yr[i] = yAbsMin;
    if (xMin > xr[i])
      xMin = xr[i];
    if (xMax < xr[i])
      xMax = xr[i];
    if (yMin > yr[i])
      yMin = yr[i];
    if (yMax < yr[i])
      yMax = yr[i];
  }
  TH2F frame("frame", "Red: T, Blue: G-L, Green: G, Black: GSL", 1, xMin, xMax, 1, yMin, yMax);
  frame.GetXaxis()->SetTitle("CPU time (sec)");
  frame.GetYaxis()->SetTitle("Accuracy");
  frame.Draw();
  TGraph gt(nt, xt, yt), gl(nl, xl, yl), gg(ng, xg, yg), gr(nr, xr, yr);
  gt.SetMarkerStyle(21);
  gt.SetLineWidth(3);
  gt.SetLineColor(kRed);
  gl.SetMarkerStyle(21);
  gl.SetLineWidth(3);
  gl.SetLineColor(kBlue);
  gg.SetMarkerStyle(21);
  gg.SetLineWidth(3);
  gg.SetLineColor(kGreen);
  gr.SetMarkerStyle(21);
  gr.SetLineWidth(3);
  gr.SetLineColor(kBlack);
  gt.Draw("PC");
  gl.Draw("PC");
  gg.Draw("PC");
  gr.Draw("PC");
  canvas.SaveAs("integralTiming.eps");
  delete[] xt;
  delete[] yt;
  delete[] xl;
  delete[] yl;
  delete[] xg;
  delete[] yg;
  delete[] xr;
  delete[] yr;

  return 0;
}
