#include "CommonTools/Statistics/interface/LinearFitErrorsIn2Coord.h"
#include "CommonTools/Statistics/interface/LinearFit.h"
#include <cmath>

float LinearFitErrorsIn2Coord::slope(
  const std::vector<float> & x, const std::vector<float> & y, int ndat, 
  const std::vector<float> & sigx, const std::vector<float> & sigy) const
{

  // scale y and sigy, compute scaled errors
  float scale = sqrt(variance(x, ndat) / variance(y, ndat));
  std::vector<float> yScaled = y;
  std::vector<float> sigyScaled = sigy;
  std::vector<float> sig(ndat);
  for (int i = 0; i != ndat; i++) {
    yScaled[i] *= scale; 
    sigyScaled[i] *= scale;
    sig[i] = sqrt(sigx[i]*sigx[i] + sigyScaled[i]*sigyScaled[i]);
  }

  // usual linear fit
  LinearFit lf;
  float fs, fi, covss, covii, covsi;
  lf.fit(x, yScaled, ndat, sig, fs, fi, covss, covii, covsi);

  // unscale result
  fs /= scale;
  return fs;

}


float LinearFitErrorsIn2Coord::intercept(
  const std::vector<float> & x, const std::vector<float> & y, int ndat, 
  const std::vector<float> & sigx, const std::vector<float> & sigy) const
{
  
  float fs = slope(x, y, ndat, sigx, sigy);
  float fi = 0;
  float sumWi = 0;
  for (int i = 0; i != ndat; i++) {
    float wi = 1./(sigy[i] + fs*fs*sigx[i]);
    fi += wi*(y[i] - fs*x[i]);
    sumWi += wi;
  }

  return fi / sumWi;

}


float 
LinearFitErrorsIn2Coord::variance(const std::vector<float> & x, int ndat) const 
{
  double m1 = 0., m2 = 0.;
  for (int i = 0; i != ndat; i++) {
    m1 += x[i];
    m2 += x[i]*x[i];
  }
  m1 /= ndat;
  m2 /= ndat;

  return float(m2 - m1*m1);

}
