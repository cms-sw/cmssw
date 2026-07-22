#include "L1Trigger/L1TGEM/interface/ME0StubFit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

double l1t::me0::reciprocal6(int n) {
  if (n >= 1 && n <= 6) {
    return RECIP6[n - 1];
  } else {
    return 0.0;  // or throw an exception
  }
}
double l1t::me0::reciprocal(int n) {
  if (n >= 1 && n <= 2047) {
    return RECIP[n - 1];
  } else {
    return 0.0;  // or throw an exception
  }
}
std::vector<double> l1t::me0::llseFit(const std::vector<double>& x, const std::vector<double>& y) {
  double xSum = 0;
  double ySum = 0;
  for (double val : x) {
    xSum += val;
  }
  for (double val : y) {
    ySum += val;
  }
  int n = x.size();
  // linear regression
  double product = 0;
  double squares = 0;
  for (int i = 0; i < n; ++i) {
    product += (n * x[i] - xSum) * (n * y[i] - ySum);
    squares += (n * x[i] - xSum) * (n * x[i] - xSum);
  }

  double m = product / squares;
  double b = (ySum - m * xSum) / n;
  double sse = 0.0;
  for (int i = 0; i < n; ++i) {
    sse += (y[i] - m * x[i] - b) * (y[i] - m * x[i] - b);
  }

  std::vector<double> fit = {m, b, sse / n};
  return fit;
}
std::vector<double> l1t::me0::vhdlExactFit(const std::vector<int>& centroids,
                                           const std::vector<bool>& validMask,
                                           bool verbose) {
  // if true not in validMask return 0, 0, 0
  if (std::find(validMask.begin(), validMask.end(), true) == validMask.end()) {
    return {0.0, 0.0, 0.0};
  }

  // remove invalid centroids
  std::vector<int> x;
  std::vector<int> y;
  for (size_t i = 0; i < centroids.size(); ++i) {
    if (validMask[i]) {
      x.push_back(i);
      y.push_back(centroids[i]);
    }
  }

  if (verbose) {
    LogTrace("ME0StubFit") << "vhdlExactFit: x=";
    for (const auto& val : x)
      LogTrace("ME0StubFit") << val << " ";
    LogTrace("ME0StubFit") << ", y=";
    for (const auto& val : y)
      LogTrace("ME0StubFit") << val << " ";
    LogTrace("ME0StubFit") << "\n";
  }

  // Stage 1
  int validCount = x.size();
  ap_uint<4> sumX = std::accumulate(x.begin(), x.end(), 0);
  ap_int<10> sumY = std::accumulate(y.begin(), y.end(), 0);

  std::vector<ap_uint<5>> nX;
  std::vector<ap_int<10>> nY;
  nX.reserve(6);
  nY.reserve(6);
  for (size_t i = 0; i < x.size(); ++i) {
    nX.push_back(validCount * x[i]);
    nY.push_back(validCount * y[i]);
  }

  // Stage 2
  std::vector<ap_int<6>> xDiff;
  std::vector<ap_int<10>> yDiff;
  xDiff.reserve(6);
  yDiff.reserve(6);
  for (size_t i = 0; i < x.size(); ++i) {
    xDiff.push_back(nX[i] - sumX);
    yDiff.push_back(nY[i] - sumY);
  }

  // Stage 3
  std::vector<ap_int<15>> product;
  std::vector<ap_int<12>> square;
  for (size_t i = 0; i < xDiff.size(); ++i) {
    product.push_back(xDiff[i] * yDiff[i]);
    square.push_back(xDiff[i] * xDiff[i]);
  }

  // Stage 4-5
  ap_int<14> sumProduct = std::accumulate(product.begin(), product.end(), 0);
  ap_int<14> sumSquare = std::accumulate(square.begin(), square.end(), 0);

  ap_fixed<14, 14, AP_RND> sumProductFixed = sumProduct;
  ap_fixed<15, 2, AP_RND> sumSquareReciprocal = l1t::me0::reciprocal(static_cast<int>(sumSquare));

  // Stage 6-7
  ap_fixed<29, 16, AP_RND> slopeTemp = sumProductFixed * sumSquareReciprocal;
  ap_fixed<10, 4, AP_RND> slope = slopeTemp;

  ap_fixed<5, 5, AP_RND> sumXFixed = sumX;

  // Stage 8
  ap_fixed<16, 10, AP_RND> slopeMult = slope * sumXFixed;

  // Stage 9
  ap_fixed<15, 8, AP_RND> slopeTimesX = slopeMult;

  // Stage 10, 11, 12
  ap_fixed<7, 7, AP_RND> sumYFixed = sumY;
  ap_fixed<16, 2, AP_RND> validCountReciprocal = l1t::me0::reciprocal6(validCount);
  ap_fixed<32, 11, AP_RND> interceptMult = validCountReciprocal * (sumYFixed - slopeTimesX);
  ap_fixed<20, 8, AP_RND> slopeS10Mult = slope * static_cast<ap_fixed<10, 4, AP_RND>>(5.0);

  ap_fixed<15, 7, AP_RND> slopeS11X5 = slopeS10Mult;

  ap_fixed<15, 7, AP_RND> slopeS12X2p5 = slopeS11X5 * static_cast<ap_fixed<15, 7, AP_RND>>(0.5);

  ap_fixed<15, 7, AP_RND> intercept = interceptMult;

  if (verbose) {
    LogTrace("ME0StubFit") << "sumY = " << sumY << "\n"
                           << "sumYFixed = " << std::format("{:.16f}", sumYFixed.to_double()) << "\n"
                           << "slopeTimesX = " << std::format("{:.16f}", slopeTimesX.to_double()) << "\n"
                           << "validCountReciprocal = " << std::format("{:.16f}", validCountReciprocal.to_double())
                           << "\n"
                           << "intercept: " << std::format("{:.16f}", intercept.to_double()) << "\n"
                           << "interceptMult: " << std::format("{:.16f}", interceptMult.to_double()) << "\n"
                           << "slopeS10Mult = " << std::format("{:.16f}", slopeS10Mult.to_double()) << "\n"
                           << "slopeS11X5 = " << std::format("{:.16f}", slopeS11X5.to_double()) << "\n"
                           << "slopeS12X2p5 = " << std::format("{:.16f}", slopeS12X2p5.to_double()) << "\n";
  }

  // Stage 13 : Output
  ap_fixed<10, 5, AP_RND> stripOut = slopeS12X2p5 + intercept;
  ap_fixed<14, 7, AP_RND> interceptOut = intercept;
  const ap_fixed<10, 4, AP_RND>& slopeOut = slope;

  if (verbose) {
    LogTrace("ME0StubFit") << "vhdlExactFit: "
                           << "slopeOut=" << std::format("{:.16f}", slopeOut.to_double())
                           << ", interceptOut=" << std::format("{:.16f}", interceptOut.to_double())
                           << ", stripOut=" << std::format("{:.16f}", stripOut.to_double());
  }

  return {slopeOut.to_double(), interceptOut.to_double(), stripOut.to_double()};
}
