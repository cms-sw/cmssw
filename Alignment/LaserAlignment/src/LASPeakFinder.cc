
#include "Alignment/LaserAlignment/interface/LASPeakFinder.h"

///
///
///
LASPeakFinder::LASPeakFinder() {
  amplitudeThreshold = 0.;  // default
}

///
/// set a profile to work on and start peak finding;
/// the pair<> will return mean/meanError (in strips);
/// offset is necessary for tob modules which are hit off-center
///
bool LASPeakFinder::FindPeakIn(const LASModuleProfile& aProfile,
                               std::pair<double, double>& result,
                               TH1D* histogram,
                               const double offset) {
  // this function will be propagated to the histo output file,
  // therefore we do some cosmetics already here
  TF1 fitFunction("fitFunction", "gaus");
  fitFunction.SetLineColor(2);
  fitFunction.SetLineWidth(2);

  std::pair<int, double> largestAmplitude(0, 0.);  // strip, amplitude
  double anAmplitude = 0.;

  // need approximate position -> cast to int
  const int approxOffset = static_cast<int>(offset);

  // expected beam position (in strips)
  const unsigned int meanPosition = 256 + approxOffset;

  // backplane "alignment hole" approx. half size (in strips)
  const unsigned int halfWindowSize = 33;

  // loop over the strips in the "alignment hole"
  // to fill the histogram
  // and determine fit parameter estimates
  double sum0 = 0.;  // this is the sum of all amplitudes
  for (unsigned int strip = meanPosition - halfWindowSize; strip < meanPosition + halfWindowSize; ++strip) {
    anAmplitude = aProfile.GetValue(strip);
    sum0 += anAmplitude;
    histogram->SetBinContent(1 + strip, anAmplitude);
    if (anAmplitude > largestAmplitude.second) {
      largestAmplitude.first = strip;
      largestAmplitude.second = anAmplitude;
    }
  }

  // loop outside the "alignment hole"
  // to determine the noise level = sqrt(variance)
  double sum1 = 0., sum2 = 0.;
  int nStrips = 0;
  for (unsigned int strip = 0; strip < 512; ++strip) {
    if (strip < meanPosition - halfWindowSize || strip > meanPosition + halfWindowSize) {
      anAmplitude = aProfile.GetValue(strip);
      sum0 += anAmplitude;
      sum1 += anAmplitude;
      sum2 += pow(anAmplitude, 2);
      nStrips++;
    }
  }

  // noise as sqrt of the amplitude variance
  const double noise = sqrt(1. / (nStrips - 1) * (sum2 - pow(sum1, 2) / nStrips));

  // empty profile?
  if (fabs(sum0) < 1.e-3) {
    std::cerr << " [LASPeakFinder::FindPeakIn] ** WARNING: Empty profile (sum=" << sum0 << ")." << std::endl;
    return false;
  }

  // no reasonable peak?
  if (largestAmplitude.second < amplitudeThreshold * noise) {
    std::cerr << " [LASPeakFinder::FindPeakIn] ** WARNING: No reasonably large peak." << std::endl;
    return false;
  }

  // prepare fit function: starting values..
  fitFunction.SetParameter(0, largestAmplitude.second);  // amp
  fitFunction.SetParameter(1, largestAmplitude.first);   // mean
  fitFunction.SetParameter(2, 3.);                       // width

  // ..and parameter limits
  fitFunction.SetParLimits(
      0, largestAmplitude.second * 0.3, largestAmplitude.second * 1.8);  // amp of the order of the peak height
  fitFunction.SetParLimits(
      1, largestAmplitude.first - 12, largestAmplitude.first + 12);  // mean around the peak maximum
  fitFunction.SetParLimits(2, 0.5, 8.);                              // reasonable width

  // fit options: Quiet / all Weights=1 / better Errors / enable fix&Bounds on parameters
  histogram->Fit(&fitFunction, "QWEB", "", largestAmplitude.first - 12, largestAmplitude.first + 12);

  // prepare output
  result.first = fitFunction.GetParameter(1);
  result.second = fitFunction.GetParError(1);

  return true;
}

///
///
///
void LASPeakFinder::SetAmplitudeThreshold(double aThreshold) { amplitudeThreshold = aThreshold; }
