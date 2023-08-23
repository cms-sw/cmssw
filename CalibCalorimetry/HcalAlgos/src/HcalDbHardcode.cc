//
// F.Ratnikov (UMd), Dec 14, 2005
//
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iostream>
#include <memory>
#include <cassert>

#include "CLHEP/Random/RandGauss.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "CalibFormats/HcalObjects/interface/HcalSiPMType.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDbHardcode::HcalDbHardcode()
    :                                              //"generic" set of conditions
      theDefaultParameters_(3.0,                   //pedestal
                            0.5,                   //pedestal width
                            {0.2, 0.2},            //gains
                            {0.0, 0.0},            //gain widths
                            0,                     //ZS threshold
                            0,                     //QIE type
                            {0.0, 0.0, 0.0, 0.0},  //QIE offsets
                            {0.9, 0.9, 0.9, 0.9},  //QIE slopes
                            125,                   //MC shape
                            105,                   //Reco shape
                            0.0,                   //photoelectronsToAnalog
                            {0.0},                 //dark current
                            {0.0},                 //noise correlation
                            0.0,                   //PF noise threshold
                            0.1                    //PF seed threshold
                            ),
      setHB_(false),
      setHE_(false),
      setHF_(false),
      setHO_(false),
      setHBUpgrade_(false),
      setHEUpgrade_(false),
      setHFUpgrade_(false),
      useHBUpgrade_(false),
      useHEUpgrade_(false),
      useHOUpgrade_(true),
      useHFUpgrade_(false),
      testHFQIE10_(false),
      testHEPlan1_(false) {}

const HcalHardcodeParameters& HcalDbHardcode::getParameters(HcalGenericDetId fId) const {
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {
    if (useHBUpgrade_ && setHBUpgrade_)
      return theHBUpgradeParameters_;
    else if (!useHBUpgrade_ && setHB_)
      return theHBParameters_;
    else
      return theDefaultParameters_;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {
    bool b_isHEPlan1 = testHEPlan1_ ? isHEPlan1(fId) : false;
    if ((useHEUpgrade_ || b_isHEPlan1) && setHEUpgrade_)
      return theHEUpgradeParameters_;
    else if (!useHEUpgrade_ && !b_isHEPlan1 && setHE_)
      return theHEParameters_;
    else
      return theDefaultParameters_;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    if (useHFUpgrade_ && setHFUpgrade_)
      return theHFUpgradeParameters_;
    else if (testHFQIE10_ && fId.isHcalDetId()) {
      HcalDetId hid(fId);
      //special mixed case for HF 2016
      if (hid.iphi() == 39 && hid.zside() == 1 &&
          (hid.depth() >= 3 || (hid.depth() == 2 && (hid.ieta() == 30 || hid.ieta() == 34))) && setHFUpgrade_)
        return theHFUpgradeParameters_;
      else if (setHF_)
        return theHFParameters_;
      else
        return theDefaultParameters_;
    } else if (!useHFUpgrade_ && setHF_)
      return theHFParameters_;
    else
      return theDefaultParameters_;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    if (setHO_)
      return theHOParameters_;
    else
      return theDefaultParameters_;
  } else
    return theDefaultParameters_;
}

const int HcalDbHardcode::getGainIndex(HcalGenericDetId fId) const {
  int index = 0;
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    HcalDetId hid(fId);
    if ((hid.ieta() > -5) && (hid.ieta() < 5))
      index = 0;
    else
      index = 1;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    HcalDetId hid(fId);
    if (hid.depth() % 2 == 1)
      index = 0;  //depths 1,3
    else if (hid.depth() % 2 == 0)
      index = 1;  //depths 2,4
  }
  return index;
}

HcalPedestal HcalDbHardcode::makePedestal(
    HcalGenericDetId fId, bool fSmear, bool eff, const HcalTopology* topo, double intlumi) {
  HcalPedestalWidth width = makePedestalWidth(fId, eff, topo, intlumi);
  float value0 = getParameters(fId).pedestal();
  if (eff) {
    //account for dark current + crosstalk
    auto sipmpar = makeHardcodeSiPMParameter(fId, topo, intlumi);
    auto sipmchar = makeHardcodeSiPMCharacteristics();
    value0 += sipmpar.getDarkCurrent() * 25. / (1. - sipmchar->getCrossTalk(sipmpar.getType()));
  }
  float value[4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value[i] = 0.0f;
      while (value[i] <= 0.0f)
        // ignore correlations, assume 10K pedestal run
        value[i] = value0 + (float)CLHEP::RandGauss::shoot(0.0, width.getWidth(i) / 100.);
    }
  }
  HcalPedestal result(fId.rawId(), value[0], value[1], value[2], value[3]);
  return result;
}

HcalPedestalWidth HcalDbHardcode::makePedestalWidth(HcalGenericDetId fId,
                                                    bool eff,
                                                    const HcalTopology* topo,
                                                    double intlumi) {
  float value = getParameters(fId).pedestalWidth();
  float width2 = value * value;
  // everything in fC

  if (eff) {
    //account for dark current + crosstalk
    auto sipmpar = makeHardcodeSiPMParameter(fId, topo, intlumi);
    auto sipmchar = makeHardcodeSiPMCharacteristics();
    //add in quadrature
    width2 += sipmpar.getDarkCurrent() * 25. / std::pow(1 - sipmchar->getCrossTalk(sipmpar.getType()), 3) *
              sipmpar.getFCByPE();
  }

  HcalPedestalWidth result(fId.rawId());
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result.setSigma(i, j, 0.0);
    }
    result.setSigma(i, i, width2);
  }
  return result;
}

HcalGain HcalDbHardcode::makeGain(HcalGenericDetId fId, bool fSmear) const {  // GeV/fC
  HcalGainWidth width = makeGainWidth(fId);
  float value0 = getParameters(fId).gain(getGainIndex(fId));
  float value[4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value[i] = 0.0f;
      while (value[i] <= 0.0f)
        value[i] = value0 + (float)CLHEP::RandGauss::shoot(0.0, width.getValue(i));
    }
  }
  HcalGain result(fId.rawId(), value[0], value[1], value[2], value[3]);
  return result;
}

HcalGainWidth HcalDbHardcode::makeGainWidth(HcalGenericDetId fId) const {  // GeV/fC
  float value = getParameters(fId).gainWidth(getGainIndex(fId));
  HcalGainWidth result(fId.rawId(), value, value, value, value);
  return result;
}

HcalPFCut HcalDbHardcode::makePFCut(HcalGenericDetId fId) const {  // GeV
  float value0 = getParameters(fId).noiseThreshold();
  float value1 = getParameters(fId).seedThreshold();
  HcalPFCut result(fId.rawId(), value0, value1);
  return result;
}

HcalZSThreshold HcalDbHardcode::makeZSThreshold(HcalGenericDetId fId) const {
  int value = getParameters(fId).zsThreshold();
  HcalZSThreshold result(fId.rawId(), value);
  return result;
}

HcalQIECoder HcalDbHardcode::makeQIECoder(HcalGenericDetId fId) const {
  HcalQIECoder result(fId.rawId());
  // slope in ADC/fC
  const HcalHardcodeParameters& param(getParameters(fId));
  for (unsigned range = 0; range < 4; range++) {
    for (unsigned capid = 0; capid < 4; capid++) {
      result.setOffset(capid, range, param.qieOffset(range));
      result.setSlope(capid, range, param.qieSlope(range));
    }
  }

  return result;
}

HcalQIEType HcalDbHardcode::makeQIEType(HcalGenericDetId fId) const {
  HcalQIENum qieType = (HcalQIENum)(getParameters(fId).qieType());
  HcalQIEType result(fId.rawId(), qieType);
  return result;
}

HcalCalibrationQIECoder HcalDbHardcode::makeCalibrationQIECoder(HcalGenericDetId fId) const {
  HcalCalibrationQIECoder result(fId.rawId());
  float lowEdges[64];
  for (int i = 0; i < 64; i++) {
    lowEdges[i] = -1.5 + i;
  }
  result.setMinCharges(lowEdges);
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape() const { return HcalQIEShape(); }

HcalMCParam HcalDbHardcode::makeMCParam(HcalGenericDetId fId) const {
  int r1bit[5];
  r1bit[0] = 9;  //  [0,9]
  int syncPhase = 0;
  r1bit[1] = 1;
  int binOfMaximum = 0;
  r1bit[2] = 4;
  float phase = -25.0f;                   // [-25.0,25.0]
  float Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                          // (offset 50nsec,  0.25ns step)
  int Iphase = Xphase;
  r1bit[3] = 8;  // [0,256] offset 50ns, .25ns step
  int timeSmearing = 0;
  r1bit[4] = 1;  //  bool

  const HcalHardcodeParameters& hparam(getParameters(fId));
  int pulseShapeID = hparam.mcShape();  // a0

  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {
    syncPhase = 1;                    // a1  bool
    binOfMaximum = 5;                 // a2
    phase = 5.0f;                     // a3  [-25.0,25.0]
    Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                      // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;
    timeSmearing = 1;  // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {
    syncPhase = 1;                    // a1  bool
    binOfMaximum = 5;                 // a2
    phase = 5.0f;                     // a3  [-25.0,25.0]
    Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                      // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;
    timeSmearing = 1;  // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    syncPhase = 1;                    // a1  bool
    binOfMaximum = 5;                 // a2
    phase = 5.0f;                     // a3  [-25.0,25.0]
    Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                      // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;
    timeSmearing = 0;  // a4

    HcalDetId cell = HcalDetId(fId);
    if (cell.ieta() == 1 && cell.iphi() == 1)
      pulseShapeID = 125;

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    syncPhase = 1;                    // a1  bool
    binOfMaximum = 3;                 // a2
    phase = 14.0f;                    // a3  [-25.0,25.0]
    Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                      // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;
    timeSmearing = 0;  // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenZDC) {
    pulseShapeID = 401;               // a0
    syncPhase = 1;                    // a1  bool
    binOfMaximum = 5;                 // a2
    phase = -4.0f;                    // a3  [-25.0,25.0]
    Xphase = (phase + 32.0f) * 4.0f;  // never change this line
                                      // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;
    timeSmearing = 0;  // a4
  }

  int rshift[7];
  rshift[0] = 0;
  for (int k = 0; k < 5; k++) {
    rshift[k + 1] = rshift[k] + r1bit[k];
  }

  int packingScheme = 1;  // a5
  unsigned int param = pulseShapeID | syncPhase << rshift[1] | (binOfMaximum << rshift[2]) | (Iphase << rshift[3]) |
                       (timeSmearing << rshift[4] | packingScheme << 27);

  HcalMCParam result(fId.rawId(), param);
  return result;
}

HcalRecoParam HcalDbHardcode::makeRecoParam(HcalGenericDetId fId) const {
  // Mostly comes from S.Kunori's macro
  int p1bit[6];

  // param1
  int containmentCorrectionFlag = 0;
  p1bit[0] = 1;  // bool
  int containmentCorrectionPreSample = 0;
  p1bit[1] = 1;                         // bool
  float phase = 13.0;                   // [-25.0,25.0]
  float Xphase = (phase + 32.0) * 4.0;  //never change this line
                                        // (offset 50nsec,  0.25ns step)
  int Iphase = Xphase;
  p1bit[2] = 8;  // [0,256]
                 // (offset 50ns, 0.25ns step
  int firstSample = 4;
  p1bit[3] = 4;  // [0,9]
  int samplesToAdd = 2;
  p1bit[4] = 4;  // [0,9]
  p1bit[5] = 9;  // [0,9]

  const HcalHardcodeParameters& hparam(getParameters(fId));
  int pulseShapeID = hparam.recoShape();  // a5

  int q2bit[10];
  //  param2.
  int useLeakCorrection = 0;
  q2bit[0] = 1;  // bool
  int LeakCorrectionID = 0;
  q2bit[1] = 4;  // [0,15]
  int correctForTimeslew = 0;
  q2bit[2] = 1;  // bool
  int timeCorrectionID = 0;
  q2bit[3] = 4;  // [0,15]
  int correctTiming = 0;
  q2bit[4] = 1;  // bool
  int firstAuxTS = 0;
  q2bit[5] = 4;  // [0,15]
  int specialCaseID = 0;
  q2bit[6] = 4;  // [0,15]
  int noiseFlaggingID = 0;
  q2bit[7] = 4;  // [0,15]
  int pileupCleaningID = 0;
  q2bit[8] = 4;  // [0,15]
  int packingScheme = 1;
  q2bit[9] = 4;

  if ((fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) ||
      (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap)) {
    //  param1.
    containmentCorrectionFlag = 1;       // p0
    containmentCorrectionPreSample = 0;  // p1
    float phase = 6.0;
    float Xphase = (phase + 32.0) * 4.0;  // never change this line
                                          //(offset 50nsec, 0.25ns step)
    Iphase = Xphase;                      // p2
    firstSample = 4;                      // p3
    samplesToAdd = 2;                     // p4

    //  param2.
    useLeakCorrection = 0;   //  q0
    LeakCorrectionID = 0;    //  q1
    correctForTimeslew = 1;  //  q2
    timeCorrectionID = 0;    //  q3
    correctTiming = 1;       //  q4
    firstAuxTS = 4;          //  q5
    specialCaseID = 0;       //  q6
    noiseFlaggingID = 1;     //  q7
    pileupCleaningID = 0;    //  q8
  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    //  param1.
    containmentCorrectionFlag = 1;       // p0
    containmentCorrectionPreSample = 0;  // p1
    float phase = 13.0;
    float Xphase = (phase + 32.0) * 4.0;  // never change this line
                                          // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;                      // p2
    firstSample = 4;                      // p3
    samplesToAdd = 4;                     // p4

    //  param2.
    useLeakCorrection = 0;   //  q0
    LeakCorrectionID = 0;    //  q1
    correctForTimeslew = 1;  //  q2
    timeCorrectionID = 0;    //  q3
    correctTiming = 1;       //  q4
    firstAuxTS = 4;          //  q5
    specialCaseID = 0;       //  q6
    noiseFlaggingID = 1;     //  q7
    pileupCleaningID = 0;    //  q8

  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    //  param1.
    containmentCorrectionFlag = 0;       // p0
    containmentCorrectionPreSample = 0;  // p1
    float phase = 13.0;
    float Xphase = (phase + 32.0) * 4.0;  // never change this line
                                          // (offset 50nsec,  0.25ns step)
    Iphase = Xphase;                      // p2
    firstSample = 2;                      // p3
    samplesToAdd = 1;                     // p4

    //  param2.
    useLeakCorrection = 0;   //  q0
    LeakCorrectionID = 0;    //  q1
    correctForTimeslew = 0;  //  q2
    timeCorrectionID = 0;    //  q3
    correctTiming = 1;       //  q4
    firstAuxTS = 1;          //  q5
    specialCaseID = 0;       //  q6
    noiseFlaggingID = 1;     //  q7
    pileupCleaningID = 0;    //  q8
  }

  // Packing parameters in two words

  int p1shift[7];
  p1shift[0] = 0;
  for (int k = 0; k < 6; k++) {
    int j = k + 1;
    p1shift[j] = p1shift[k] + p1bit[k];
    //     cout<<"  j= "<<j<<"  shift "<< p1shift[j]<<endl;
  }
  int param1 = 0;
  param1 = containmentCorrectionFlag | (containmentCorrectionPreSample << p1shift[1]) | (Iphase << p1shift[2]) |
           (firstSample << p1shift[3]) | (samplesToAdd << p1shift[4]) | (pulseShapeID << p1shift[5]);

  int q2shift[10];
  q2shift[0] = 0;
  for (int k = 0; k < 9; k++) {
    int j = k + 1;
    q2shift[j] = q2shift[k] + q2bit[k];
    //  cout<<"  j= "<<j<<"  shift "<< q2shift[j]<<endl;
  }
  int param2 = 0;
  param2 = useLeakCorrection | (LeakCorrectionID << q2shift[1]) | (correctForTimeslew << q2shift[2]) |
           (timeCorrectionID << q2shift[3]) | (correctTiming << q2shift[4]) | (firstAuxTS << q2shift[5]) |
           (specialCaseID << q2shift[6]) | (noiseFlaggingID << q2shift[7]) | (pileupCleaningID << q2shift[8]) |
           (packingScheme << q2shift[9]);

  HcalRecoParam result(fId.rawId(), param1, param2);

  return result;
}

HcalTimingParam HcalDbHardcode::makeTimingParam(HcalGenericDetId fId) const {
  int nhits = 0;
  float phase = 0.0;
  float rms = 0.0;
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {
    nhits = 4;
    phase = 4.5;
    rms = 6.5;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {
    nhits = 4;
    phase = 9.3;
    rms = 7.8;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    nhits = 4;
    phase = 8.6;
    rms = 2.3;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    nhits = 4;
    phase = 12.4;
    rms = 12.29;
  }
  HcalTimingParam result(fId.rawId(), nhits, phase, rms);

  return result;
}

#define EMAP_NHBHECR 9
#define EMAP_NHFCR 3
#define EMAP_NHOCR 4
#define EMAP_NFBR 8
#define EMAP_NFCH 3
#define EMAP_NHTRS 3
#define EMAP_NHSETS 4
#define EMAP_NTOPBOT 2
#define EMAP_NHTRSHO 4
#define EMAP_NHSETSHO 3

std::unique_ptr<HcalDcsMap> HcalDbHardcode::makeHardcodeDcsMap() const {
  HcalDcsMapAddons::Helper dcs_map_helper;
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalBarrel, -16, 1, 1),
                                 HcalDcsDetId(HcalDcsBarrel, -1, 1, HcalDcsDetId::HV, 2));
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalForward, -41, 3, 1),
                                 HcalDcsDetId(HcalDcsForward, -1, 1, HcalDcsDetId::DYN8, 1));
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalForward, -26, 25, 2),
                                 HcalDcsDetId(HcalDcsForward, -1, 7, HcalDcsDetId::HV, 1));
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalBarrel, -15, 68, 1),
                                 HcalDcsDetId(HcalDcsBarrel, -1, 18, HcalDcsDetId::HV, 3));
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalOuter, -14, 1, 4),
                                 HcalDcsDetId(HcalDcsOuter, -2, 2, HcalDcsDetId::HV, 4));
  dcs_map_helper.mapGeomId2DcsId(HcalDetId(HcalForward, 41, 71, 2),
                                 HcalDcsDetId(HcalDcsForward, 1, 4, HcalDcsDetId::DYN8, 3));
  return std::make_unique<HcalDcsMap>(dcs_map_helper);
}

std::unique_ptr<HcalElectronicsMap> HcalDbHardcode::makeHardcodeMap(const std::vector<HcalGenericDetId>& cells) const {
  static const int kUTCAMask = 0x4000000;         //set bit 26 for uTCA version
  static const int kLinearIndexMax = 0x7FFFF;     //19 bits
  static const int kTriggerBitMask = 0x02000000;  //2^25
  uint32_t counter = 0;
  uint32_t counterTrig = 0;
  HcalElectronicsMapAddons::Helper emapHelper;
  for (const auto& fId : cells) {
    if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel ||
        fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap ||
        fId.genericSubdet() == HcalGenericDetId::HcalGenForward ||
        fId.genericSubdet() == HcalGenericDetId::HcalGenOuter || fId.genericSubdet() == HcalGenericDetId::HcalGenZDC) {
      ++counter;
      assert(counter < kLinearIndexMax);
      uint32_t raw = counter;
      raw |= kUTCAMask;
      HcalElectronicsId elId(raw);
      emapHelper.mapEId2chId(elId, fId);
    } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenTriggerTower) {
      ++counterTrig;
      assert(counterTrig < kLinearIndexMax);
      uint32_t raw = counterTrig;
      raw |= kUTCAMask | kTriggerBitMask;
      HcalElectronicsId elId(raw);
      emapHelper.mapEId2tId(elId, fId);
    }
  }
  return std::make_unique<HcalElectronicsMap>(emapHelper);
}

std::unique_ptr<HcalFrontEndMap> HcalDbHardcode::makeHardcodeFrontEndMap(
    const std::vector<HcalGenericDetId>& cells) const {
  HcalFrontEndMapAddons::Helper emapHelper;
  std::stringstream mystream;
  std::string detector[5] = {"XX", "HB", "HE", "HO", "HF"};
  for (const auto& fId : cells) {
    if (fId.isHcalDetId()) {
      HcalDetId id = HcalDetId(fId.rawId());
      HcalSubdetector subdet = id.subdet();
      int ieta = id.ietaAbs();
      int iside = id.zside();
      int iphi = id.iphi();
      std::string det, rbx;
      int irm(0);
      char tempbuff[30];
      char sidesign = (iside == -1) ? 'M' : 'P';
      if (subdet == HcalBarrel || subdet == HcalEndcap) {
        det = detector[subdet];
        irm = (iphi + 1) % 4 + 1;
        int iwedge(0);
        if (ieta >= 21 && (irm == 1 || irm == 3))
          iwedge = (iphi + 1 + irm + 1) / 4;
        else
          iwedge = (iphi + irm + 1) / 4;
        if (iwedge > 18)
          iwedge -= 18;
        sprintf(tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, iwedge, '\0');
        mystream << tempbuff;
        rbx = mystream.str();
        mystream.str("");
        emapHelper.loadObject(id, irm, rbx);
      } else if (subdet == HcalForward) {
        det = detector[subdet];
        int hfphi(0);
        if ((iside == 1 && ieta == 40) || (iside == -1 && ieta == 41)) {
          irm = ((iphi + 1) / 2) % 36 + 1;
          hfphi = ((iphi + 1) / 6) % 12 + 1;
        } else {
          irm = (iphi + 1) / 2;
          hfphi = (iphi - 1) / 6 + 1;
        }
        irm = (irm - 1) % 3 + 1;
        sprintf(tempbuff, "%s%c%2.2i%c", det.c_str(), sidesign, hfphi, '\0');
        mystream << tempbuff;
        rbx = mystream.str();
        mystream.str("");
        emapHelper.loadObject(id, irm, rbx);
      } else if (subdet == HcalOuter) {
        det = detector[subdet];
        int ring(0), sector(0);
        if (ieta <= 4)
          ring = 0;
        else if (ieta >= 5 && ieta <= 10)
          ring = 1;
        else
          ring = 2;
        for (int i = -2; i < iphi; i += 6)
          sector++;
        if (sector > 12)
          sector = 1;
        irm = ((iphi + 1) / 2) % 6 + 1;
        if (ring != 0 && sector % 2 != 0)
          sector++;
        if (ring == 0)
          sprintf(tempbuff, "%s%i%2.2d", det.c_str(), ring, sector);
        else
          sprintf(tempbuff, "%s%i%c%2.2d", det.c_str(), ring, sidesign, sector);
        mystream << tempbuff;
        rbx = mystream.str();
        mystream.str("");
        emapHelper.loadObject(id, irm, rbx);
      }
    }
  }
  return std::make_unique<HcalFrontEndMap>(emapHelper);
}

int HcalDbHardcode::getLayersInDepth(int ieta, int depth, const HcalTopology* topo) {
  //check for cached value
  auto eta_depth_pair = std::make_pair(ieta, depth);
  auto nLayers = theLayersInDepths_.find(eta_depth_pair);
  if (nLayers != theLayersInDepths_.end()) {
    return nLayers->second;
  } else {
    std::vector<int> segmentation;
    topo->getDepthSegmentation(ieta, segmentation);
    //assume depth segmentation vector is sorted
    int nLayersInDepth = std::distance(std::lower_bound(segmentation.begin(), segmentation.end(), depth),
                                       std::upper_bound(segmentation.begin(), segmentation.end(), depth));
    theLayersInDepths_.insert(std::make_pair(eta_depth_pair, nLayersInDepth));
    return nLayersInDepth;
  }
}

bool HcalDbHardcode::isHEPlan1(HcalGenericDetId fId) const {
  if (fId.isHcalDetId()) {
    HcalDetId hid(fId);
    //special mixed case for HE 2017
    if (hid.zside() == 1 && (hid.iphi() == 63 || hid.iphi() == 64 || hid.iphi() == 65 || hid.iphi() == 66))
      return true;
  }
  return false;
}

HcalSiPMParameter HcalDbHardcode::makeHardcodeSiPMParameter(HcalGenericDetId fId,
                                                            const HcalTopology* topo,
                                                            double intlumi) {
  // SiPMParameter defined for each DetId the following quantities:
  //  SiPM type, PhotoElectronToAnalog, Dark Current, two auxiliary words
  //  (the second of those containing float noise correlation coefficient
  //  These numbers come from some measurements done with SiPMs
  // rule for type: cells with >4 layers use larger device (3.3mm diameter), otherwise 2.8mm
  HcalSiPMType theType = HcalNoSiPM;
  double thePe2fC = getParameters(fId).photoelectronsToAnalog();
  double theDC = getParameters(fId).darkCurrent(0, intlumi);
  double theNoiseCN = getParameters(fId).noiseCorrelation(0);
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {
    if (useHBUpgrade_) {
      HcalDetId hid(fId);
      int nLayersInDepth = getLayersInDepth(hid.ietaAbs(), hid.depth(), topo);
      if (nLayersInDepth > 4) {
        theType = HcalHBHamamatsu2;
        theDC = getParameters(fId).darkCurrent(1, intlumi);
        theNoiseCN = getParameters(fId).noiseCorrelation(1);
      } else {
        theType = HcalHBHamamatsu1;
        theDC = getParameters(fId).darkCurrent(0, intlumi);
        theNoiseCN = getParameters(fId).noiseCorrelation(0);
      }
    } else
      theType = HcalHPD;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {
    if (useHEUpgrade_ || (testHEPlan1_ && isHEPlan1(fId))) {
      HcalDetId hid(fId);
      int nLayersInDepth = getLayersInDepth(hid.ietaAbs(), hid.depth(), topo);
      if (nLayersInDepth > 4) {
        theType = HcalHEHamamatsu2;
        theDC = getParameters(fId).darkCurrent(1, intlumi);
        theNoiseCN = getParameters(fId).noiseCorrelation(1);
      } else {
        theType = HcalHEHamamatsu1;
        theDC = getParameters(fId).darkCurrent(0, intlumi);
        theNoiseCN = getParameters(fId).noiseCorrelation(0);
      }
    } else
      theType = HcalHPD;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    if (useHOUpgrade_)
      theType = HcalHOHamamatsu;
    else
      theType = HcalHPD;
  }

  return HcalSiPMParameter(fId.rawId(), theType, thePe2fC, theDC, 0, (float)theNoiseCN);
}

std::unique_ptr<HcalSiPMCharacteristics> HcalDbHardcode::makeHardcodeSiPMCharacteristics() const {
  // SiPMCharacteristics are constants for each type of SiPM:
  // Type, # of pixels, 3 parameters for non-linearity, cross talk parameter, ..
  // Obtained from data sheet and measurements
  // types (in order): HcalHOZecotek=1, HcalHOHamamatsu, HcalHEHamamatsu1, HcalHEHamamatsu2, HcalHBHamamatsu1, HcalHBHamamatsu2, HcalHPD
  HcalSiPMCharacteristicsAddons::Helper sipmHelper;
  for (unsigned ip = 0; ip < theSiPMCharacteristics_.size(); ++ip) {
    auto& ps = theSiPMCharacteristics_[ip];
    sipmHelper.loadObject(ip + 1,
                          ps.getParameter<int>("pixels"),
                          ps.getParameter<double>("nonlin1"),
                          ps.getParameter<double>("nonlin2"),
                          ps.getParameter<double>("nonlin3"),
                          ps.getParameter<double>("crosstalk"),
                          0,
                          0);
  }
  return std::make_unique<HcalSiPMCharacteristics>(sipmHelper);
}

HcalTPChannelParameter HcalDbHardcode::makeHardcodeTPChannelParameter(HcalGenericDetId fId) const {
  // For each detId parameters for trigger primitive
  // mask for channel validity and self trigger information, fine grain
  // bit information and auxiliary words
  uint32_t bitInfo = ((44 << 16) | 30);
  return HcalTPChannelParameter(fId.rawId(), 0, bitInfo, 0, 0);
}

void HcalDbHardcode::makeHardcodeTPParameters(HcalTPParameters& tppar) const {
  // Parameters for a given TP algorithm:
  // FineGrain Algorithm Version for HBHE, ADC threshold fof TDC mask of HF,
  // TDC mask for HF, Self Trigger bits, auxiliary words
  tppar.loadObject(0, 0, 0xFFFFFFFFFFFFFFFF, 0, 0, 0);
}
