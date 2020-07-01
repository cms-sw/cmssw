//
// Negative energy filter parameters developed by Alexander Toropin for Run 2 data
//
#include <memory>



#include "CondTools/Hcal/interface/make_HBHENegativeEFilter.h"

#define make_poly /**/                                                                    \
  std::vector<std::vector<double> > coeffs;                                               \
  coeffs.reserve(6);                                                                      \
  coeffs.push_back(std::vector<double>(pol41, pol41 + sizeof(pol41) / sizeof(pol41[0]))); \
  coeffs.push_back(std::vector<double>(pol42, pol42 + sizeof(pol42) / sizeof(pol42[0]))); \
  coeffs.push_back(std::vector<double>(pol31, pol31 + sizeof(pol31) / sizeof(pol31[0]))); \
  coeffs.push_back(std::vector<double>(pol32, pol32 + sizeof(pol32) / sizeof(pol32[0]))); \
  coeffs.push_back(std::vector<double>(pol11, pol11 + sizeof(pol11) / sizeof(pol11[0]))); \
  coeffs.push_back(std::vector<double>(pol12, pol12 + sizeof(pol12) / sizeof(pol12[0]))); \
  const std::vector<double> lim(limits, limits + 5);                                      \
  return PiecewiseScalingPolynomial(coeffs, lim)

static PiecewiseScalingPolynomial a1_HB() {  // version 23.02.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.7261, -0.03124, 0.001177, -1.349e-05, -5.512e-08};
  const double pol42[5] = {0.4151, 0.006921, -0.0003291, 4.897e-06, -2.42e-08};
  const double pol31[4] = {0.469, -0.00151, 4.307e-06, -4.697e-09};
  const double pol32[4] = {0.5264, -0.001902, 4.613e-06, -3.808e-09};
  const double pol11[2] = {0.2648, -1.942e-05};
  const double pol12[2] = {0.2413, -3.638e-07};
  const double limits[5] = {28., 60., 190., 435., 1330.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_HB() {  // version 23.02.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.3148, -0.03012, 0.001738, -4.53e-05, 4.414e-07};
  const double pol42[5] = {0.1315, -0.00103, 1.192e-05, -6.723e-08, 3.843e-10};
  const double pol31[4] = {0.1108, -5.343e-05, -1.06e-06, 3.75e-09};
  const double pol32[4] = {0.1045, -0.0001121, 1.443e-07, -7.224e-11};
  const double pol11[2] = {0.07919, -1.252e-05};
  const double pol12[2] = {0.06529, -3.825e-06};
  const double limits[5] = {23., 65., 190., 850., 1640.};
  make_poly;
}

// ------------------------ 17-20 ----------------------------------------
static PiecewiseScalingPolynomial a1_1720() {  // version 28.03.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.6129, -0.00765, -0.0008141, 5.724e-05, -9.377e-07};
  const double pol42[5] = {0.4033, 0.00678, -0.000312, 4.534e-06, -2.197e-08};
  const double pol31[4] = {0.4603, -0.001526, 4.85e-06, -6.446e-09};
  const double pol32[4] = {0.4849, -0.001505, 3.307e-06, -2.461e-09};
  const double pol11[2] = {0.261, -2.086e-05};
  const double pol12[1] = {0.2354};
  const double limits[5] = {23., 68., 190., 410., 1320.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_1720() {  // version 28.03.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.313, -0.02747, 0.001404, -3.232e-05, 2.748e-07};
  const double pol42[5] = {0.1519, -0.003287, 7.798e-05, -9.27e-07, 4.455e-09};
  const double pol31[4] = {0.1079, -0.0002475, 2.859e-07, 7.703e-10};
  const double pol32[4] = {0.09147, -0.0001035, 1.268e-07, -5.747e-11};
  const double pol11[2] = {0.06536, -8.462e-06};
  const double pol12[1] = {0.05337};
  const double limits[5] = {27., 68., 190., 1050., 1550.};
  make_poly;
}

// ------------------------ 21-23 ----------------------------------------
static PiecewiseScalingPolynomial a1_2123() {  // version 28.03.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.4908, 0.01378, -0.002152, 9.244e-05, -1.27e-06};
  const double pol42[5] = {0.3983, 0.00721, -0.000334, 4.999e-06, -2.52e-08};
  const double pol31[4] = {0.4746, -0.001923, 8.029e-06, -1.411e-08};
  const double pol32[4] = {0.4848, -0.001503, 3.339e-06, -2.518e-09};
  const double pol11[2] = {0.264, -2.491e-05};
  const double pol12[2] = {0.2545, -1.539e-05};
  const double limits[5] = {23., 68., 190., 515., 1240.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_2123() {  // version 28.03.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.2886, -0.01573, 0.0003735, 3.986e-06, -1.723e-07};
  const double pol42[5] = {0.1724, -0.003205, 6.239e-05, -5.989e-07, 2.346e-09};
  const double pol31[4] = {0.1302, -0.0004984, 2.021e-06, -3.34e-09};
  const double pol32[4] = {0.104, -0.0001241, 1.579e-07, -7.438e-11};
  const double pol11[2] = {0.07527, -1.186e-05};
  const double pol12[1] = {0.05917};
  const double limits[5] = {23., 68., 190., 1000., 1380.};
  make_poly;
}

// ------------------------ 24-25 ----------------------------------------
static PiecewiseScalingPolynomial a1_2425() {  // version 28.03.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.4595, 0.01439, -0.001756, 6.716e-05, -8.552e-07};
  const double pol42[5] = {0.3636, 0.01037, -0.0004347, 6.349e-06, -3.167e-08};
  const double pol31[4] = {0.4698, -0.001775, 6.638e-06, -1.061e-08};
  const double pol32[4] = {0.4854, -0.001519, 3.321e-06, -2.483e-09};
  const double pol11[2] = {0.2568, -2.584e-05};
  const double pol12[2] = {0.2282, -2.334e-06};
  const double limits[5] = {23., 68., 190., 425., 1250.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_2425() {  // version 28.03.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.2922, -0.009065, -0.0002295, 2.453e-05, -4.171e-07};
  const double pol42[5] = {0.2008, -0.003443, 5.083e-05, -3.064e-07, 4.728e-10};
  const double pol31[4] = {0.1533, -0.0007735, 3.571e-06, -6.432e-09};
  const double pol32[4] = {0.111, -0.000138, 1.746e-07, -8.205e-11};
  const double pol11[2] = {0.07719, -1.123e-05};
  const double pol12[2] = {0.0637, -1.798e-06};
  const double limits[5] = {24., 68., 220., 975., 1450.};
  make_poly;
}

// ------------------------ 26-27 ----------------------------------------
static PiecewiseScalingPolynomial a1_2627() {  // version 28.03.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.4404, 0.01173, -0.001344, 4.862e-05, -5.896e-07};
  const double pol42[5] = {0.3483, 0.01019, -0.000416, 5.989e-06, -2.946e-08};
  const double pol31[4] = {0.4545, -0.0016, 5.716e-06, -8.879e-09};
  const double pol32[4] = {0.4794, -0.001485, 3.206e-06, -2.369e-09};
  const double pol11[2] = {0.2531, -2.403e-05};
  const double pol12[2] = {0.2376, -9.953e-06};
  const double limits[5] = {23., 68., 190., 515., 1200.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_2627() {  // version 28.03.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.3104, -0.01182, 0.0001538, 4.599e-06, -9.634e-08};
  const double pol42[5] = {0.2105, -0.003244, 2.824e-05, 1.422e-07, -2.243e-09};
  const double pol31[4] = {0.1587, -0.0008399, 3.787e-06, -6.468e-09};
  const double pol32[4] = {0.1107, -0.0001325, 1.632e-07, -7.479e-11};
  const double pol11[2] = {0.07933, -1.265e-05};
  const double pol12[2] = {0.06511, -2.133e-06};
  const double limits[5] = {20., 68., 215., 1000., 1350.};
  make_poly;
}

// ------------------------ 28 ----------------------------------------
static PiecewiseScalingPolynomial a1_28() {  // version 28.03.2014
  // this function is made from profile TS5/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.4552, 0.00774, -0.0007652, 2.166e-05, -2.04e-07};
  const double pol42[5] = {0.4447, 0.001959, -0.0001573, 2.587e-06, -1.341e-08};
  const double pol31[4] = {0.4364, -0.0009454, 9.873e-07, 1.515e-09};
  const double pol32[4] = {0.4726, -0.00135, 2.811e-06, -2.047e-09};
  const double pol11[2] = {0.2569, -2.931e-05};
  const double pol12[2] = {0.2345, -8.457e-06};
  const double limits[5] = {23., 68., 190., 515., 1150.};
  make_poly;
}

static PiecewiseScalingPolynomial a2_28() {  // version 28.03.2014
  // this function is made from profile TS6/TS4 vs TS4, MultiJet
  const double pol41[5] = {0.3477, -0.002766, -0.0003802, 1.631e-05, -1.871e-07};
  const double pol42[5] = {0.3404, -0.006505, 8.54e-05, -4.002e-07, -4.685e-11};
  const double pol31[4] = {0.2295, -0.001221, 4.778e-06, -7.352e-09};
  const double pol32[4] = {0.1589, -0.0002629, 3.0e-07, -1.249e-10};
  const double pol11[2] = {0.08305, -1.098e-05};
  const double pol12[2] = {0.07318, -3.478e-06};
  const double limits[5] = {23., 68., 190., 970., 1300.};
  make_poly;
}

std::unique_ptr<HBHENegativeEFilter> make_HBHENegativeEFilter() {
  // |ieta| limits for the "a1" and "a2" shapes
  const unsigned etaLim[5] = {17, 21, 24, 26, 28};
  const std::vector<uint32_t> lim(etaLim, etaLim + sizeof(etaLim) / sizeof(etaLim[0]));

  // Each element of the shape vector corresponds to
  // one of the |ieta| region defined by "lim". Element 0
  // corresponds to |ieta| < lim[0], element 1 corresponds
  // to lim[0] <= |ieta| < lim[1], etc.
  std::vector<PiecewiseScalingPolynomial> a1vec;
  a1vec.reserve(6);
  a1vec.push_back(a1_HB());
  a1vec.push_back(a1_1720());
  a1vec.push_back(a1_2123());
  a1vec.push_back(a1_2425());
  a1vec.push_back(a1_2627());
  a1vec.push_back(a1_28());

  std::vector<PiecewiseScalingPolynomial> a2vec;
  a2vec.reserve(6);
  a2vec.push_back(a2_HB());
  a2vec.push_back(a2_1720());
  a2vec.push_back(a2_2123());
  a2vec.push_back(a2_2425());
  a2vec.push_back(a2_2627());
  a2vec.push_back(a2_28());

  // The "negative energy" discriminant. The first element
  // of the pair is the collected charge in the [firstTS, lastTS]
  // window, and the second element is the value of the discriminant
  // (the "pass" region is above the line). Interpolation between
  // the points is linear. Extrapolation below the point with the
  // lowest charge and above the point with the highest charge is
  // constant, from the nearest point.
  //
  // The discriminant is applied only if the charge collected in
  // the window is at least "minChargeThreshold".
  //
  std::vector<std::pair<double, double> > cut;
  cut.reserve(4);
  cut.push_back(std::make_pair(0.0, -25.0));
  cut.push_back(std::make_pair(400.0, -45.0));
  cut.push_back(std::make_pair(1500.0, -135.0));
  cut.push_back(std::make_pair(15000.0, -1000.0));

  const double minChargeThreshold = 20.0;
  const unsigned firstTS = 4;
  const unsigned lastTS = 6;

  return std::make_unique<HBHENegativeEFilter>(
      a1vec, a2vec, lim, cut, minChargeThreshold, firstTS, lastTS);
}
