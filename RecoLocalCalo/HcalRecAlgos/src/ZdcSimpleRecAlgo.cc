#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <algorithm>  // for "max"
#include <iostream>
#include <cmath>

constexpr double MaximumFractionalError = 0.0005;  // 0.05% error allowed from this source

ZdcSimpleRecAlgo::ZdcSimpleRecAlgo(
    bool correctForTimeslew, bool correctForPulse, float phaseNS, int recoMethod, int lowGainOffset, double lowGainFrac)
    : recoMethod_(recoMethod),
      correctForTimeslew_(correctForTimeslew),
      correctForPulse_(correctForPulse),
      phaseNS_(phaseNS),
      lowGainOffset_(lowGainOffset),
      lowGainFrac_(lowGainFrac) {}

ZdcSimpleRecAlgo::ZdcSimpleRecAlgo(int recoMethod) : recoMethod_(recoMethod), correctForTimeslew_(false) {}

void ZdcSimpleRecAlgo::initPulseCorr(int toadd, const HcalTimeSlew* hcalTimeSlew_delay) {
  if (correctForPulse_) {
    pulseCorr_ = std::make_unique<HcalPulseContainmentCorrection>(
        toadd, phaseNS_, false, MaximumFractionalError, hcalTimeSlew_delay);
  }
}
//static float timeshift_ns_zdc(float wpksamp);

namespace ZdcSimpleRecAlgoImpl {
  template <class Digi, class RecHit>
  inline RecHit reco1(const Digi& digi,
                      const HcalCoder& coder,
                      const HcalCalibrations& calibs,
                      const std::vector<unsigned int>& myNoiseTS,
                      const std::vector<unsigned int>& mySignalTS,
                      int lowGainOffset,
                      double lowGainFrac,
                      bool slewCorrect,
                      const HcalPulseContainmentCorrection* corr,
                      HcalTimeSlew::BiasSetting slewFlavor) {
    CaloSamples tool;
    coder.adc2fC(digi, tool);
    int ifirst = mySignalTS[0];
    int n = mySignalTS.size();
    double ampl = 0;
    int maxI = -1;
    double maxA = -1e10;
    double ta = 0;
    double fc_ampl = 0;
    double lowGEnergy = 0;
    double TempLGAmp = 0;
    // TS increment for regular energy to lowGainEnergy
    // Signal in higher TS (effective "low Gain") has a fraction of the whole signal
    // This constant for fC --> GeV is dervied from 2010 PbPb analysis of single neutrons
    // assumed similar fraction for EM and HAD sections
    // this variable converts from current assumed TestBeam values for fC--> GeV
    // to the lowGain TS region fraction value (based on 1N Had, assume EM same response)
    // regular energy
    for (int i = ifirst; i < tool.size() && i < n + ifirst; i++) {
      int capid = digi[i].capid();
      ta = (tool[i] - calibs.pedestal(capid));  // pedestal subtraction
      fc_ampl += ta;
      ta *= calibs.respcorrgain(capid);  // fC --> GeV
      ampl += ta;
      if (ta > maxA) {
        maxA = ta;
        maxI = i;
      }
    }
    // calculate low Gain Energy (in 2010 PbPb, signal TS 4,5,6, lowGain TS: 6,7,8)
    int topLowGain = 10;
    if ((n + ifirst + lowGainOffset) <= 10) {
      topLowGain = n + ifirst + lowGainOffset;
    } else {
      topLowGain = 10;
    }
    for (int iLG = (ifirst + lowGainOffset); iLG < tool.size() && iLG < topLowGain; iLG++) {
      int capid = digi[iLG].capid();
      TempLGAmp = (tool[iLG] - calibs.pedestal(capid));  // pedestal subtraction
      TempLGAmp *= calibs.respcorrgain(capid);           // fC --> GeV
      TempLGAmp *= lowGainFrac;                          // TS (signalRegion) --> TS (lowGainRegion)
      lowGEnergy += TempLGAmp;
    }
    double time = -9999;
    // Time based on regular energy (lowGainEnergy signal assumed to happen at same Time)
    ////Cannot calculate time value with max ADC sample at first or last position in window....
    if (maxI == 0 || maxI == (tool.size() - 1)) {
      LogDebug("HCAL Pulse") << "ZdcSimpleRecAlgo::reco1 :"
                             << " Invalid max amplitude position, "
                             << " max Amplitude: " << maxI << " first: " << ifirst << " last: " << (tool.size() - 1)
                             << std::endl;
    } else {
      int capid = digi[maxI - 1].capid();
      double Energy0 = ((tool[maxI - 1]) * calibs.respcorrgain(capid));
      // if any of the energies used in the weight are negative, make them 0 instead
      // these are actually QIE values, not energy
      if (Energy0 < 0) {
        Energy0 = 0.;
      }
      capid = digi[maxI].capid();
      double Energy1 = ((tool[maxI]) * calibs.respcorrgain(capid));
      if (Energy1 < 0) {
        Energy1 = 0.;
      }
      capid = digi[maxI + 1].capid();
      double Energy2 = ((tool[maxI + 1]) * calibs.respcorrgain(capid));
      if (Energy2 < 0) {
        Energy2 = 0.;
      }
      //
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
      if (corr != nullptr) {
        // Apply phase-based amplitude correction:
        ampl *= corr->getCorrection(fc_ampl);
      }
    }
    return RecHit(digi.id(), ampl, time, lowGEnergy);
  }
}  // namespace ZdcSimpleRecAlgoImpl

namespace ZdcSimpleRecAlgoImpl {
  template <class Digi, class RecHit>
  inline RecHit reco2(const Digi& digi,
                      const HcalCoder& coder,
                      const HcalCalibrations& calibs,
                      const std::vector<unsigned int>& myNoiseTS,
                      const std::vector<unsigned int>& mySignalTS,
                      int lowGainOffset,
                      double lowGainFrac,
                      bool slewCorrect,
                      const HcalPulseContainmentCorrection* corr,
                      HcalTimeSlew::BiasSetting slewFlavor) {
    CaloSamples tool;
    coder.adc2fC(digi, tool);
    // Reads noiseTS and signalTS from database
    int ifirst = mySignalTS[0];
    //    int n = mySignalTS.size();
    double ampl = 0;
    int maxI = -1;
    double maxA = -1e10;
    double ta = 0;
    double fc_ampl = 0;
    double lowGEnergy = 0;
    double TempLGAmp = 0;
    //  TS increment for regular energy to lowGainEnergy
    // Signal in higher TS (effective "low Gain") has a fraction of the whole signal
    // This constant for fC --> GeV is dervied from 2010 PbPb analysis of single neutrons
    // assumed similar fraction for EM and HAD sections
    // this variable converts from current assumed TestBeam values for fC--> GeV
    // to the lowGain TS region fraction value (based on 1N Had, assume EM same response)
    double Allnoise = 0;
    int noiseslices = 0;
    int CurrentTS = 0;
    double noise = 0;
    // regular energy (both use same noise)
    for (unsigned int iv = 0; iv < myNoiseTS.size(); ++iv) {
      CurrentTS = myNoiseTS[iv];
      if (CurrentTS >= digi.size())
        continue;
      Allnoise += tool[CurrentTS];
      noiseslices++;
    }
    if (noiseslices != 0) {
      noise = (Allnoise) / double(noiseslices);
    } else {
      noise = 0;
    }
    for (unsigned int ivs = 0; ivs < mySignalTS.size(); ++ivs) {
      CurrentTS = mySignalTS[ivs];
      if (CurrentTS >= digi.size())
        continue;
      int capid = digi[CurrentTS].capid();
      //       if(noise<0){
      //       // flag hit as having negative noise, and don't subtract anything, because
      //       // it will falsely increase the energy
      //          noisefactor=0.;
      //       }
      ta = tool[CurrentTS] - noise;
      fc_ampl += ta;
      ta *= calibs.respcorrgain(capid);  // fC --> GeV
      ampl += ta;
      if (ta > maxA) {
        maxA = ta;
        maxI = CurrentTS;
      }
    }
    // calculate low Gain Energy (in 2010 PbPb, signal TS 4,5,6, lowGain TS: 6,7,8)
    for (unsigned int iLGvs = 0; iLGvs < mySignalTS.size(); ++iLGvs) {
      CurrentTS = mySignalTS[iLGvs] + lowGainOffset;
      if (CurrentTS >= digi.size())
        continue;
      int capid = digi[CurrentTS].capid();
      TempLGAmp = tool[CurrentTS] - noise;
      TempLGAmp *= calibs.respcorrgain(capid);  // fC --> GeV
      TempLGAmp *= lowGainFrac;                 // TS (signalRegion) --> TS (lowGainRegion)
      lowGEnergy += TempLGAmp;
    }
    //    if(ta<0){
    //      // flag hits that have negative energy
    //    }

    double time = -9999;
    // Time based on regular energy (lowGainEnergy signal assumed to happen at same Time)
    ////Cannot calculate time value with max ADC sample at first or last position in window....
    if (maxI == 0 || maxI == (tool.size() - 1)) {
      LogDebug("HCAL Pulse") << "ZdcSimpleRecAlgo::reco2 :"
                             << " Invalid max amplitude position, "
                             << " max Amplitude: " << maxI << " first: " << ifirst << " last: " << (tool.size() - 1)
                             << std::endl;
    } else {
      int capid = digi[maxI - 1].capid();
      double Energy0 = ((tool[maxI - 1]) * calibs.respcorrgain(capid));
      // if any of the energies used in the weight are negative, make them 0 instead
      // these are actually QIE values, not energy
      if (Energy0 < 0) {
        Energy0 = 0.;
      }
      capid = digi[maxI].capid();
      double Energy1 = ((tool[maxI]) * calibs.respcorrgain(capid));
      if (Energy1 < 0) {
        Energy1 = 0.;
      }
      capid = digi[maxI + 1].capid();
      double Energy2 = ((tool[maxI + 1]) * calibs.respcorrgain(capid));
      if (Energy2 < 0) {
        Energy2 = 0.;
      }
      //
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
      if (corr != nullptr) {
        // Apply phase-based amplitude correction:
        ampl *= corr->getCorrection(fc_ampl);
      }
    }
    return RecHit(digi.id(), ampl, time, lowGEnergy);
  }
}  // namespace ZdcSimpleRecAlgoImpl

ZDCRecHit ZdcSimpleRecAlgo::reconstruct(const ZDCDataFrame& digi,
                                        const std::vector<unsigned int>& myNoiseTS,
                                        const std::vector<unsigned int>& mySignalTS,
                                        const HcalCoder& coder,
                                        const HcalCalibrations& calibs) const {
  if (recoMethod_ == 1)
    return ZdcSimpleRecAlgoImpl::reco1<ZDCDataFrame, ZDCRecHit>(
        digi, coder, calibs, myNoiseTS, mySignalTS, lowGainOffset_, lowGainFrac_, false, nullptr, HcalTimeSlew::Fast);
  if (recoMethod_ == 2)
    return ZdcSimpleRecAlgoImpl::reco2<ZDCDataFrame, ZDCRecHit>(
        digi, coder, calibs, myNoiseTS, mySignalTS, lowGainOffset_, lowGainFrac_, false, nullptr, HcalTimeSlew::Fast);

  edm::LogError("ZDCSimpleRecAlgoImpl::reconstruct, recoMethod was not declared");
  throw cms::Exception("ZDCSimpleRecoAlgos::exiting process");
}
