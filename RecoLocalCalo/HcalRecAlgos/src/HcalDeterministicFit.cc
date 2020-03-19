#include <iostream>
#include <cmath>
#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"

constexpr float HcalDeterministicFit::invGpar[3];
constexpr float HcalDeterministicFit::negThresh[2];
constexpr int HcalDeterministicFit::HcalRegion[2];
constexpr float HcalDeterministicFit::rCorr[2];
constexpr float HcalDeterministicFit::rCorrSiPM[2];

using namespace std;

HcalDeterministicFit::HcalDeterministicFit() {}

HcalDeterministicFit::~HcalDeterministicFit() {}

void HcalDeterministicFit::init(HcalTimeSlew::ParaSource tsParam,
                                HcalTimeSlew::BiasSetting bias,
                                bool iApplyTimeSlew,
                                double respCorr) {
  fTimeSlew_ = tsParam;
  fTimeSlewBias_ = bias;
  applyTimeSlew_ = iApplyTimeSlew;
  frespCorr_ = respCorr;
}

constexpr float HcalDeterministicFit::landauFrac[];
// Landau function integrated in 1 ns intervals
//Landau pulse shape from https://indico.cern.ch/event/345283/contribution/3/material/slides/0.pdf
//Landau turn on by default at left edge of time slice
// normalized to 1 on [0,10000]
void HcalDeterministicFit::getLandauFrac(float tStart, float tEnd, float &sum) const {
  if (std::abs(tStart - tEnd - tsWidth) < 0.1f) {
    sum = 0.f;
    return;
  }
  sum = landauFrac[int(ceil(tStart + tsWidth))];
  return;
}

constexpr float HcalDeterministicFit::siPM205Frac[];
void HcalDeterministicFit::get205Frac(float tStart, float tEnd, float &sum) const {
  if (std::abs(tStart - tEnd - tsWidth) < 0.1f) {
    sum = 0.f;
    return;
  }
  sum = siPM205Frac[int(ceil(tStart + tsWidth))];
  return;
}

constexpr float HcalDeterministicFit::siPM206Frac[];
void HcalDeterministicFit::get206Frac(float tStart, float tEnd, float &sum) const {
  if (std::abs(tStart - tEnd - tsWidth) < 0.1f) {
    sum = 0.f;
    return;
  }
  sum = siPM206Frac[int(ceil(tStart + tsWidth))];
  return;
}

constexpr float HcalDeterministicFit::siPM207Frac[];
void HcalDeterministicFit::get207Frac(float tStart, float tEnd, float &sum) const {
  if (std::abs(tStart - tEnd - tsWidth) < 0.1f) {
    sum = 0.f;
    return;
  }
  sum = siPM207Frac[int(ceil(tStart + tsWidth))];
  return;
}

void HcalDeterministicFit::getFrac(float tStart, float tEnd, float &sum, FType fType) const {
  switch (fType) {
    case shape205:
      get205Frac(tStart, tEnd, sum);
      break;
    case shape206:
      get206Frac(tStart, tEnd, sum);
      break;
    case shape207:
      get207Frac(tStart, tEnd, sum);
      break;
    case shapeLandau:
      getLandauFrac(tStart, tEnd, sum);
      break;
  }
}

void HcalDeterministicFit::phase1Apply(const HBHEChannelInfo &channelData,
                                       float &reconstructedEnergy,
                                       float &reconstructedTime,
                                       const HcalTimeSlew *hcalTimeSlew_delay) const {
  unsigned int soi = channelData.soi();

  std::vector<double> corrCharge;
  corrCharge.reserve(channelData.nSamples());
  std::vector<double> inputCharge;
  inputCharge.reserve(channelData.nSamples());
  std::vector<double> inputPedestal;
  inputPedestal.reserve(channelData.nSamples());
  std::vector<double> inputNoise;
  inputNoise.reserve(channelData.nSamples());

  double gainCorr = 0;
  double respCorr = 0;

  for (unsigned int ip = 0; ip < channelData.nSamples(); ip++) {
    double charge = channelData.tsRawCharge(ip);
    double ped = channelData.tsPedestal(ip);
    double noise = channelData.tsPedestalWidth(ip);
    double gain = channelData.tsGain(ip);

    gainCorr = gain;
    inputCharge.push_back(charge);
    inputPedestal.push_back(ped);
    inputNoise.push_back(noise);
  }

  fPedestalSubFxn_.calculate(inputCharge, inputPedestal, inputNoise, corrCharge, soi, channelData.nSamples());

  if (fTimeSlew_ == 0)
    respCorr = 1.0;
  else if (fTimeSlew_ == 1)
    channelData.hasTimeInfo() ? respCorr = rCorrSiPM[0] : respCorr = rCorr[0];
  else if (fTimeSlew_ == 2)
    channelData.hasTimeInfo() ? respCorr = rCorrSiPM[1] : respCorr = rCorr[1];
  else if (fTimeSlew_ == 3)
    respCorr = frespCorr_;

  float tsShift3, tsShift4, tsShift5;
  tsShift3 = 0.f, tsShift4 = 0.f, tsShift5 = 0.f;

  if (applyTimeSlew_) {
    tsShift3 = hcalTimeSlew_delay->delay(inputCharge[soi - 1], fTimeSlew_, fTimeSlewBias_, !channelData.hasTimeInfo());
    tsShift4 = hcalTimeSlew_delay->delay(inputCharge[soi], fTimeSlew_, fTimeSlewBias_, !channelData.hasTimeInfo());
    tsShift5 = hcalTimeSlew_delay->delay(inputCharge[soi + 1], fTimeSlew_, fTimeSlewBias_, !channelData.hasTimeInfo());
  }

  float ch3, ch4, ch5, i3, n3, nn3, i4, n4, i5, n5;
  ch4 = 0.f, i3 = 0.f, n3 = 0.f, nn3 = 0.f, i4 = 0.f, n4 = 0.f, i5 = 0.f, n5 = 0.f;

  FType fType;
  if (channelData.hasTimeInfo() && channelData.recoShape() == 205)
    fType = shape205;
  else if (channelData.hasTimeInfo() && channelData.recoShape() == 206)
    fType = shape206;
  else if (channelData.hasTimeInfo() && channelData.recoShape() == 207)
    fType = shape207;
  else
    fType = shapeLandau;

  getFrac(-tsShift3, -tsShift3 + tsWidth, i3, fType);
  getFrac(-tsShift3 + tsWidth, -tsShift3 + tsWidth * 2, n3, fType);
  getFrac(-tsShift3 + tsWidth * 2, -tsShift3 + tsWidth * 3, nn3, fType);

  getFrac(-tsShift4, -tsShift4 + tsWidth, i4, fType);
  getFrac(-tsShift4 + tsWidth, -tsShift4 + tsWidth * 2, n4, fType);

  getFrac(-tsShift5, -tsShift5 + tsWidth, i5, fType);
  getFrac(-tsShift5 + tsWidth, -tsShift5 + tsWidth * 2, n5, fType);

  if (i3 != 0 && i4 != 0 && i5 != 0) {
    ch3 = corrCharge[soi - 1] / i3;
    ch4 = (i3 * corrCharge[soi] - n3 * corrCharge[soi - 1]) / (i3 * i4);
    ch5 = (n3 * n4 * corrCharge[soi - 1] - i4 * nn3 * corrCharge[soi - 1] - i3 * n4 * corrCharge[soi] +
           i3 * i4 * corrCharge[soi + 1]) /
          (i3 * i4 * i5);

    if (ch3 < negThresh[0]) {
      ch3 = negThresh[0];
      ch4 = corrCharge[soi] / i4;
      ch5 = (i4 * corrCharge[soi + 1] - n4 * corrCharge[soi]) / (i4 * i5);
    }
    if (ch5 < negThresh[0] && ch4 > negThresh[1]) {
      double ratio = (corrCharge[soi] - ch3 * i3) / (corrCharge[soi + 1] - negThresh[0] * i5);
      if (ratio < 5 && ratio > 0.5) {
        double invG = invGpar[0] + invGpar[1] * std::sqrt(2 * std::log(invGpar[2] / ratio));
        float iG = 0.f;
        getFrac(-invG, -invG + tsWidth, iG, fType);
        if (iG != 0) {
          ch4 = (corrCharge[soi] - ch3 * n3) / (iG);
          tsShift4 = invG;
        }
      }
    }
  }

  if (ch4 < 1) {
    ch4 = 0.f;
  }

  reconstructedEnergy = ch4 * gainCorr * respCorr;
  reconstructedTime = tsShift4;
}
