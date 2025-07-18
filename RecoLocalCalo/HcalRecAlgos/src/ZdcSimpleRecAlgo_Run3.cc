#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo_Run3.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>  // for "max"
#include <iostream>
#include <cmath>

#include <Eigen/Dense>  // for TemplateFit Method

// #include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

ZdcSimpleRecAlgo_Run3::ZdcSimpleRecAlgo_Run3(int recoMethod) : recoMethod_(recoMethod) {}

void ZdcSimpleRecAlgo_Run3::initCorrectionMethod(const int method, const int ZdcSection) {
  correctionMethod_[ZdcSection] = method;
};

// Template fit is a linear combination of timeslices in a digi assuming there is a potential charge peak at each Bx
// and the charges of the Ts follwing a peak are consistent with the chargeRatios
void ZdcSimpleRecAlgo_Run3::initTemplateFit(const std::vector<unsigned int>& bxTs,
                                            const std::vector<double>& chargeRatios,
                                            const int nTs,
                                            const int ZdcSection) {
  int nRatios = chargeRatios.size();
  int nBx = bxTs.size();
  int nCols = nBx + 1;
  double val = 0;
  int index = 0;
  int timeslice = 0;
  nTs_ = nTs;
  Eigen::MatrixXf a(nTs, nCols);
  for (int j = 0; j < nBx; j++) {
    timeslice = bxTs.at(j);
    for (int i = 0; i < nTs; i++) {
      val = 0;
      index = i - timeslice;
      if (index >= 0 && index < nRatios)
        val = chargeRatios.at(index);
      a(i, j) = val;
    }
  }
  for (int i = 0; i < nTs; i++)
    a(i, nBx) = 1;
  Eigen::MatrixXf b = a.transpose() * a;
  if (std::fabs(b.determinant()) < 1E-8) {
    templateFitValid_[ZdcSection] = false;
    return;
  }
  templateFitValid_[ZdcSection] = true;
  Eigen::MatrixXf tfMatrix;
  tfMatrix = b.inverse() * a.transpose();
  for (int i = 0; i < nTs; i++)
    templateFitValues_[ZdcSection].push_back(tfMatrix.coeff(1, i));
}

void ZdcSimpleRecAlgo_Run3::initRatioSubtraction(const float ratio, const float frac, const int ZdcSection) {
  ootpuRatio_[ZdcSection] = ratio;
  ootpuFrac_[ZdcSection] = frac;
}

// helper functions for pedestal subtraction and noise calculations
namespace zdchelper {

  inline double subPedestal(const float charge, const float ped, const float width) {
    if (charge - ped > width)
      return (charge - ped);
    else
      return (0);
  }

  // gets the noise with a check if the ratio of energy0 / energy1 > ootpuRatio
  // energy1 is noiseTS , energy0 is noiseTs-1
  inline double getNoiseOOTPURatio(const float energy0,
                                   const float energy1,
                                   const float ootpuRatio,
                                   const float ootpuFrac) {
    if (energy0 >= ootpuRatio * energy1 || ootpuRatio < 0)
      return (ootpuFrac * energy1);
    else
      return (energy1);
  }

}  // namespace zdchelper

ZDCRecHit ZdcSimpleRecAlgo_Run3::reco0(const QIE10DataFrame& digi,
                                       const HcalCoder& coder,
                                       const HcalCalibrations& calibs,
                                       const HcalPedestal& effPeds,
                                       const std::vector<unsigned int>& myNoiseTS,
                                       const std::vector<unsigned int>& mySignalTS) const {
  CaloSamples tool;
  coder.adc2fC(digi, tool);
  // Reads noiseTS and signalTS from database
  int ifirst = mySignalTS[0];
  double ampl = 0;
  int maxI = -1;
  double maxA = -1e10;
  double ta = 0;
  double energySOIp1 = 0;
  double ratioSOIp1 = -1.0;
  double chargeWeightedTime = -99.0;

  double Allnoise = 0;
  int noiseslices = 0;
  int CurrentTS = 0;
  double noise = 0;
  int digi_size = digi.samples();
  HcalZDCDetId cell = digi.id();
  int zdcsection = cell.section();

  // determining noise
  for (unsigned int iv = 0; iv < myNoiseTS.size(); ++iv) {
    CurrentTS = myNoiseTS[iv];
    int capid = digi[CurrentTS].capid();
    float ped = effPeds.getValue(capid);
    float width = effPeds.getWidth(capid);
    float gain = calibs.respcorrgain(capid);
    if (CurrentTS >= digi_size)
      continue;
    float energy1 = zdchelper::subPedestal(tool[CurrentTS], ped, width) * gain;
    float energy0 = 0;
    if (CurrentTS > 0) {
      capid = digi[CurrentTS - 1].capid();
      ped = effPeds.getValue(capid);
      width = effPeds.getWidth(capid);
      gain = calibs.respcorrgain(capid);
      energy0 = zdchelper::subPedestal(tool[CurrentTS - 1], ped, width) * gain;
    }
    Allnoise += zdchelper::getNoiseOOTPURatio(energy0, energy1, ootpuRatio_.at(zdcsection), ootpuFrac_.at(zdcsection));
    noiseslices++;
  }
  if (noiseslices != 0) {
    noise = (Allnoise) / double(noiseslices);
  }

  // determining signal energy and max Ts
  for (unsigned int ivs = 0; ivs < mySignalTS.size(); ++ivs) {
    CurrentTS = mySignalTS[ivs];
    if (CurrentTS >= digi_size)
      continue;
    float energy1 = -1;
    int capid = digi[CurrentTS].capid();
    // float ped = calibs.pedestal(capid);
    float ped = effPeds.getValue(capid);
    float width = effPeds.getWidth(capid);
    float gain = calibs.respcorrgain(capid);
    float energy0 = std::max(0.0, zdchelper::subPedestal(tool[CurrentTS], ped, width)) * gain;
    if (CurrentTS < digi_size - 1) {
      capid = digi[CurrentTS].capid();
      ped = effPeds.getValue(capid);
      width = effPeds.getWidth(capid);
      gain = calibs.respcorrgain(capid);
      energy1 = std::max(0.0, zdchelper::subPedestal(tool[CurrentTS + 1], ped, width)) * gain;
    }
    ta = energy0 - noise;
    if (ta > 0)
      ampl += ta;

    if (ta > maxA) {
      ratioSOIp1 = (energy0 > 0 && energy1 > 0) ? energy0 / energy1 : -1.0;
      maxA = ta;
      maxI = CurrentTS;
    }
  }

  // determine energy if using Template Fit method
  if (correctionMethod_.at(zdcsection) == 1 && templateFitValid_.at(zdcsection)) {
    double energy = 0;
    for (int iv = 0; iv < nTs_; iv++) {
      int capid = digi[iv].capid();
      float ped = effPeds.getValue(capid);
      float width = effPeds.getWidth(capid);
      float gain = calibs.respcorrgain(capid);
      if (iv >= digi_size)
        continue;
      energy += zdchelper::subPedestal(tool[iv], ped, width) * (templateFitValues_.at(zdcsection)).at(iv) * gain;
    }
    ampl = std::max(0.0, energy);
  }

  double time = -9999;
  // Time based on regular energy
  ////Cannot calculate time value with max ADC sample at first or last position in window....
  if (maxI == 0 || maxI == (tool.size() - 1)) {
    LogDebug("HCAL Pulse") << "ZdcSimpleRecAlgo::reco2 :"
                           << " Invalid max amplitude position, "
                           << " max Amplitude: " << maxI << " first: " << ifirst << " last: " << (tool.size() - 1)
                           << std::endl;
  } else {
    int capid = digi[maxI - 1].capid();
    double Energy0 = std::max(0.0, ((tool[maxI - 1]) * calibs.respcorrgain(capid)));

    capid = digi[maxI].capid();
    double Energy1 = std::max(0.0, ((tool[maxI]) * calibs.respcorrgain(capid)));
    capid = digi[maxI + 1].capid();
    double Energy2 = std::max(0.0, ((tool[maxI + 1]) * calibs.respcorrgain(capid)));

    double TSWeightEnergy = ((maxI - 1) * Energy0 + maxI * Energy1 + (maxI + 1) * Energy2);
    double EnergySum = Energy0 + Energy1 + Energy2;
    double AvgTSPos = 0.;
    if (EnergySum != 0)
      AvgTSPos = TSWeightEnergy / EnergySum;
    // If time is zero, set it to the "nonsensical" -99
    // Time should be between 75ns and 175ns (Timeslices 3-7)
    if (AvgTSPos == 0) {
      time = -99;
    } else {
      time = (AvgTSPos * 25.0);
    }
  }

  // find energy for signal TS + 1
  for (unsigned int ivs = 0; ivs < mySignalTS.size(); ++ivs) {
    CurrentTS = mySignalTS[ivs] + 1;
    if (CurrentTS >= digi_size)
      continue;
    int capid = digi[CurrentTS].capid();
    // ta = tool[CurrentTS] - noise;
    ta = tool[CurrentTS];
    ta *= calibs.respcorrgain(capid);  // fC --> GeV
    if (ta > 0)
      energySOIp1 += ta;
  }

  double tmp_energy = 0;
  double tmp_TSWeightedEnergy = 0;
  for (int ts = 0; ts < digi_size; ++ts) {
    int capid = digi[ts].capid();

    // max sure there are no negative values in time calculation
    ta = std::max(0.0, tool[ts]);
    ta *= calibs.respcorrgain(capid);  // fC --> GeV
    if (ta > 0) {
      tmp_energy += ta;
      tmp_TSWeightedEnergy += (ts)*ta;
    }
  }

  if (tmp_energy > 0)
    chargeWeightedTime = (tmp_TSWeightedEnergy / tmp_energy) * 25.0;
  auto rh = ZDCRecHit(digi.id(), ampl, time, -99);
  rh.setEnergySOIp1(energySOIp1);

  if (maxI >= 0 && maxI < tool.size()) {
    float tmp_tdctime = 0;
    int le_tdc = digi[maxI].le_tdc();
    // TDC error codes will be 60=-1, 61 = -2, 62 = -3, 63 = -4
    if (le_tdc >= 60)
      tmp_tdctime = -1 * (le_tdc - 59);
    else
      tmp_tdctime = maxI * 25. + (le_tdc / 2.0);
    rh.setTDCtime(tmp_tdctime);
  }

  rh.setChargeWeightedTime(chargeWeightedTime);
  rh.setRatioSOIp1(ratioSOIp1);
  return rh;
}

ZDCRecHit ZdcSimpleRecAlgo_Run3::reconstruct(const QIE10DataFrame& digi,
                                             const std::vector<unsigned int>& myNoiseTS,
                                             const std::vector<unsigned int>& mySignalTS,
                                             const HcalCoder& coder,
                                             const HcalCalibrations& calibs,
                                             const HcalPedestal& effPeds) const {
  return ZdcSimpleRecAlgo_Run3::reco0(digi, coder, calibs, effPeds, myNoiseTS, mySignalTS);

  edm::LogError("ZDCSimpleRecAlgoImpl::reconstruct, recoMethod was not declared");
  throw cms::Exception("ZDCSimpleRecoAlgos::exiting process");
}