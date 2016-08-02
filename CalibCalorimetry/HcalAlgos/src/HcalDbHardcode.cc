//
// F.Ratnikov (UMd), Dec 14, 2005
//
#include <vector>
#include <string> 
#include <cmath> 
#include <sstream>
#include <iostream>

#include "CLHEP/Random/RandGauss.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDbHardcode::HcalDbHardcode()
: theDefaultParameters_(3.0,0.5,{0.2,0.2},{0.0,0.0},0,{0.0,0.0,0.0,0.0},{0.9,0.9,0.9,0.9},125,105), //"generic" set of conditions
  setHB_(false), setHE_(false), setHF_(false), setHO_(false), 
  setHBUpgrade_(false), setHEUpgrade_(false), setHFUpgrade_(false), 
  useHBUpgrade_(false), useHEUpgrade_(false), useHFUpgrade_(false), testHFQIE10_(false)
{
}

const HcalHardcodeParameters& HcalDbHardcode::getParameters(HcalGenericDetId fId){
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel){
    if(useHBUpgrade_ && setHBUpgrade_) return theHBUpgradeParameters_;
    else if(!useHBUpgrade_ && setHB_) return theHBParameters_;
    else return theDefaultParameters_;
  }
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap){
    if(useHEUpgrade_ && setHEUpgrade_) return theHEUpgradeParameters_;
    else if(!useHEUpgrade_ && setHE_) return theHEParameters_;
    else return theDefaultParameters_;
  }
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward){
    if(useHFUpgrade_ && setHFUpgrade_) return theHFUpgradeParameters_;
	else if(testHFQIE10_ && fId.isHcalDetId()){
        HcalDetId hid(fId);
        //special mixed case for HF 2016
        if(hid.iphi()==39 && hid.zside()==1 && (hid.depth()>=3 || (hid.depth()==2 && (hid.ieta()==30 || hid.ieta()==34))) && setHFUpgrade_) return theHFUpgradeParameters_;
        else if(setHF_) return theHFParameters_;
        else return theDefaultParameters_;
    }
    else if(!useHFUpgrade_ && setHF_) return theHFParameters_;
    else return theDefaultParameters_;
  }
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter){
    if(setHO_) return theHOParameters_;
    else return theDefaultParameters_;
  }
  else return theDefaultParameters_;
}

const int HcalDbHardcode::getGainIndex(HcalGenericDetId fId){
  int index = 0;
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    HcalDetId hid(fId);
    if ((hid.ieta() > -5) && (hid.ieta() < 5)) index = 0;
    else index = 1;
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    HcalDetId hid(fId);
    if (hid.depth() == 1) index = 0;
    else if (hid.depth() == 2) index = 1;
  }
  return index;
}

HcalPedestal HcalDbHardcode::makePedestal (HcalGenericDetId fId, bool fSmear) {
  HcalPedestalWidth width = makePedestalWidth (fId);
  float value0 = getParameters(fId).pedestal();
  float value [4] = {value0,value0,value0,value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value[i] = 0.0f;
      while (value [i] <= 0.0f)
	// ignore correlations, assume 10K pedestal run 
	value [i] = value0 + (float)CLHEP::RandGauss::shoot (0.0, width.getWidth (i) / 100.);
    }
  }
  HcalPedestal result (fId.rawId (), 
               value[0], value[1], value[2], value[3]
               );
  return result;
}

HcalPedestalWidth HcalDbHardcode::makePedestalWidth (HcalGenericDetId fId) {
  float value = getParameters(fId).pedestalWidth();
  // everything in fC

  HcalPedestalWidth result (fId.rawId ());
  float width2 = value*value;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result.setSigma (i, j, 0.0);
    }
    result.setSigma (i, i, width2);
  } 
  return result;
}

HcalGain HcalDbHardcode::makeGain (HcalGenericDetId fId, bool fSmear) { // GeV/fC
  HcalGainWidth width = makeGainWidth (fId);
  float value0 = getParameters(fId).gain(getGainIndex(fId));
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value[i] = 0.0f;
      while (value [i] <= 0.0f) value [i] = value0 + (float)CLHEP::RandGauss::shoot (0.0, width.getValue (i)); 
    }
  }
  HcalGain result (fId.rawId (), value[0], value[1], value[2], value[3]);
  return result;
}

HcalGainWidth HcalDbHardcode::makeGainWidth (HcalGenericDetId fId) { // GeV/fC
  float value = getParameters(fId).gainWidth(getGainIndex(fId));
  HcalGainWidth result (fId.rawId (), value, value, value, value);
  return result;
}

HcalQIECoder HcalDbHardcode::makeQIECoder (HcalGenericDetId fId) {
  HcalQIECoder result (fId.rawId ());
  // slope in ADC/fC
  const HcalHardcodeParameters& param(getParameters(fId));
  for (unsigned range = 0; range < 4; range++) {
    for (unsigned capid = 0; capid < 4; capid++) {
      result.setOffset (capid, range, param.qieOffset(range));
      result.setSlope (capid, range, param.qieSlope(range));
    }
  }

  return result;
}

HcalQIEType HcalDbHardcode::makeQIEType (HcalGenericDetId fId) {
  HcalQIENum qieType = (HcalQIENum)(getParameters(fId).qieType());
  HcalQIEType result(fId.rawId(),qieType);
  return result;
}

HcalCalibrationQIECoder HcalDbHardcode::makeCalibrationQIECoder (HcalGenericDetId fId) {
  HcalCalibrationQIECoder result (fId.rawId ());
  float lowEdges [64];
  for (int i = 0; i < 64; i++) { lowEdges[i] = -1.5 + i; }
  result.setMinCharges (lowEdges);
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape () {

  //  std::cout << " !!! HcalDbHardcode::makeQIEShape " << std::endl; 

  return HcalQIEShape ();
}


HcalMCParam HcalDbHardcode::makeMCParam (HcalGenericDetId fId) {

  int r1bit[5];
                           r1bit[0] = 9;     //  [0,9]
  int syncPhase    = 0;    r1bit[1] = 1;
  int binOfMaximum = 0;    r1bit[2] = 4;
  float phase      = -25.0f;                  // [-25.0,25.0]
  float Xphase     = (phase + 32.0f) * 4.0f;   // never change this line 
                                             // (offset 50nsec,  0.25ns step)
  int Iphase = Xphase;     r1bit[3] = 8;     // [0,256] offset 50ns, .25ns step
  int timeSmearing = 0;    r1bit[4] = 1;     //  bool

  const HcalHardcodeParameters& hparam(getParameters(fId));
  int pulseShapeID = hparam.mcShape(); // a0
 
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) { 

    syncPhase    = 1;                      // a1  bool
    binOfMaximum = 5;                      // a2
    phase        = 5.0f;                    // a3  [-25.0,25.0]
    Xphase       = (phase + 32.0f) * 4.0f;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 1;                      // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) { 

    syncPhase    = 1;                      // a1  bool
    binOfMaximum = 5;                      // a2
    phase        = 5.0f;                    // a3  [-25.0,25.0]
    Xphase       = (phase + 32.0f) * 4.0f;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 1;                      // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {

    syncPhase    = 1;                      // a1  bool
    binOfMaximum = 5;                      // a2
    phase        = 5.0f;                    // a3  [-25.0,25.0]
    Xphase       = (phase + 32.0f) * 4.0f;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a4

    HcalDetId cell = HcalDetId(fId);
    if (cell.ieta() == 1 && cell.iphi() == 1) pulseShapeID = 125;

  }
  
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) { 

    syncPhase    = 1;                      // a1  bool
    binOfMaximum = 3;                      // a2
    phase        = 14.0f;                   // a3  [-25.0,25.0]
    Xphase       = (phase + 32.0f) * 4.0f;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a4
 
  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenZDC) { 

    pulseShapeID = 401;                    // a0
    syncPhase    = 1;                      // a1  bool
    binOfMaximum = 5;                      // a2
    phase        = -4.0f;                   // a3  [-25.0,25.0]
    Xphase       = (phase + 32.0f) * 4.0f;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a4
 
  }


  int rshift[7];
  rshift[0]=0;
  for(int k=0; k<5; k++) {
    rshift[k+1]=rshift[k]+r1bit[k];
  }

  int packingScheme  = 1; // a5
  unsigned int param = pulseShapeID |
    syncPhase<<rshift[1]            |
    (binOfMaximum << rshift[2])     |
    (Iphase << rshift[3])           |
    (timeSmearing << rshift[4] | packingScheme << 27);
 
  HcalMCParam result(fId.rawId(), param);
  return result;

}

HcalRecoParam HcalDbHardcode::makeRecoParam (HcalGenericDetId fId) {

  // Mostly comes from S.Kunori's macro 
  int p1bit[6];
  
  // param1 
  int containmentCorrectionFlag = 0;       p1bit[0]=1;   // bool
  int containmentCorrectionPreSample = 0;  p1bit[1]=1;   // bool
  float phase  = 13.0;                                  // [-25.0,25.0]
  float Xphase = (phase + 32.0) * 4.0;     //never change this line 
                                           // (offset 50nsec,  0.25ns step)
  int Iphase = Xphase;                     p1bit[2]=8;   // [0,256]  
                                           // (offset 50ns, 0.25ns step
  int firstSample  = 4;                    p1bit[3]=4;   // [0,9]
  int samplesToAdd = 2;                    p1bit[4]=4;   // [0,9]
                                           p1bit[5]=9;   // [0,9]

  const HcalHardcodeParameters& hparam(getParameters(fId));
  int pulseShapeID = hparam.recoShape(); // a5

  int q2bit[10];
  //  param2.
  int useLeakCorrection  = 0;              q2bit[0]=1;   // bool
  int LeakCorrectionID   = 0;              q2bit[1]=4;   // [0,15]
  int correctForTimeslew = 0;              q2bit[2]=1;   // bool
  int timeCorrectionID   = 0;              q2bit[3]=4;   // [0,15]
  int correctTiming      = 0;              q2bit[4]=1;   // bool
  int firstAuxTS         = 0;              q2bit[5]=4;   // [0,15]
  int specialCaseID      = 0;              q2bit[6]=4;   // [0,15]
  int noiseFlaggingID    = 0;              q2bit[7]=4;   // [0,15]
  int pileupCleaningID   = 0;              q2bit[8]=4;   // [0,15]
  int packingScheme      = 1;              q2bit[9]=4;
    

  if((fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) || 
     (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap)) {
    //  param1.
    containmentCorrectionFlag = 1;         // p0
    containmentCorrectionPreSample = 0;    // p1
    float phase  = 25.0;
    float Xphase = (phase + 32.0) * 4.0;   // never change this line 
                                           //(offset 50nsec, 0.25ns step)
    Iphase       = Xphase;                 // p2
    firstSample  = 4;                      // p3
    samplesToAdd = 2;                      // p4
    
    //  param2.
    useLeakCorrection  = 0;                //  q0
    LeakCorrectionID   = 0;                //  q1
    correctForTimeslew = 1;                //  q2
    timeCorrectionID   = 0;                //  q3
    correctTiming      = 1;                //  q4
    firstAuxTS         = 4;                //  q5
    specialCaseID      = 0;                //  q6
    noiseFlaggingID    = 1;                //  q7
    pileupCleaningID   = 0;                //  q8
  } 


  else if(fId.genericSubdet() == HcalGenericDetId::HcalGenOuter ) {
    //  param1.
    containmentCorrectionFlag = 1;         // p0
    containmentCorrectionPreSample = 0;    // p1
    float  phase  = 13.0;
    float  Xphase = (phase + 32.0) * 4.0;  // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;                 // p2
    firstSample  = 4;                      // p3
    samplesToAdd = 4;                      // p4

    //  param2.
    useLeakCorrection  = 0;                //  q0
    LeakCorrectionID   = 0;                //  q1
    correctForTimeslew = 1;                //  q2
    timeCorrectionID   = 0;                //  q3
    correctTiming      = 1;                //  q4
    firstAuxTS         = 4;                //  q5
    specialCaseID      = 0;                //  q6
    noiseFlaggingID    = 1;                //  q7
    pileupCleaningID   = 0;                //  q8

  }
  else if(fId.genericSubdet() == HcalGenericDetId::HcalGenForward ) {
    //  param1.
    containmentCorrectionFlag = 0;         // p0
    containmentCorrectionPreSample = 0;    // p1
    float  phase = 13.0;
    float  Xphase = (phase + 32.0) * 4.0;  // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;                 // p2
    firstSample  = 2;                      // p3
    samplesToAdd = 1;                      // p4
    
    //  param2.
    useLeakCorrection  = 0;                //  q0
    LeakCorrectionID   = 0;                //  q1
    correctForTimeslew = 0;                //  q2
    timeCorrectionID   = 0;                //  q3
    correctTiming      = 1;                //  q4
    firstAuxTS         = 1;                //  q5
    specialCaseID      = 0;                //  q6
    noiseFlaggingID    = 1;                //  q7
    pileupCleaningID   = 0;                //  q8
  } 
  

  // Packing parameters in two words

  int p1shift[7]; p1shift[0] = 0;
  for(int k = 0; k < 6; k++) {
    int j = k + 1;
    p1shift[j] = p1shift[k] + p1bit[k];
    //     cout<<"  j= "<<j<<"  shift "<< p1shift[j]<<endl;
  }
  int param1 = 0;
  param1 = containmentCorrectionFlag               | 
    (containmentCorrectionPreSample << p1shift[1]) | 
    (Iphase                         << p1shift[2]) | 
    (firstSample                    << p1shift[3]) | 
    (samplesToAdd                   << p1shift[4]) | 
    (pulseShapeID                   << p1shift[5]) ;
  
  int q2shift[10]; q2shift[0] = 0;
  for(int k = 0; k < 9; k++) {
    int j = k + 1;
    q2shift[j] = q2shift[k] + q2bit[k];
    //  cout<<"  j= "<<j<<"  shift "<< q2shift[j]<<endl;
  }  
  int param2 = 0;
  param2 = useLeakCorrection           |
    (LeakCorrectionID   << q2shift[1]) | 
    (correctForTimeslew << q2shift[2]) |
    (timeCorrectionID   << q2shift[3]) | 
    (correctTiming      << q2shift[4]) | 
    (firstAuxTS         << q2shift[5]) |
    (specialCaseID      << q2shift[6]) | 
    (noiseFlaggingID    << q2shift[7]) | 
    (pileupCleaningID   << q2shift[8]) |
    (packingScheme      << q2shift[9]) ;
  
  HcalRecoParam result(fId.rawId(), param1, param2);

  return result;
}

HcalTimingParam HcalDbHardcode::makeTimingParam (HcalGenericDetId fId) {
  int nhits = 0;
  float phase = 0.0;
  float rms = 0.0;
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {nhits=4; phase = 4.5; rms = 6.5;}
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {nhits=4;phase = 9.3; rms = 7.8;}
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {nhits=4;phase = 8.6; rms = 2.3;}
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {nhits=4;phase = 12.4; rms = 12.29;}
  HcalTimingParam result(fId.rawId(), nhits,phase, rms);

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

void HcalDbHardcode::makeHardcodeDcsMap(HcalDcsMap& dcs_map) {
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalBarrel, -16, 1, 1), 
			  HcalDcsDetId(HcalDcsBarrel, -1, 1, HcalDcsDetId::HV, 2));
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalForward, -41, 3, 1), 
			  HcalDcsDetId(HcalDcsForward, -1, 1, HcalDcsDetId::DYN8, 1));
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalForward, -26, 25, 2), 
			  HcalDcsDetId(HcalDcsForward, -1, 7, HcalDcsDetId::HV, 1));
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalBarrel, -15, 68, 1), 
			  HcalDcsDetId(HcalDcsBarrel, -1, 18, HcalDcsDetId::HV, 3));
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalOuter, -14, 1, 4), 
			  HcalDcsDetId(HcalDcsOuter, -2, 2, HcalDcsDetId::HV, 4));
  dcs_map.mapGeomId2DcsId(HcalDetId(HcalForward, 41, 71, 2), 
			  HcalDcsDetId(HcalDcsForward, 1, 4, HcalDcsDetId::DYN8, 3));
}

void HcalDbHardcode::makeHardcodeMap(HcalElectronicsMap& emap) {

  /* HBHE crate numbering */
  int hbhecrate[EMAP_NHBHECR]={0,1,4,5,10,11,14,15,17};
  /* HF crate numbering */
  int hfcrate[EMAP_NHFCR]={2,9,12};
  /* HO crate numbering */
  int hocrate[EMAP_NHOCR]={3,7,6,13};
  /* HBHE FED numbering of DCCs */
  int fedhbhenum[EMAP_NHBHECR][2]={{702,703},{704,705},{700,701},
				   {706,707},{716,717},{708,709},
				   {714,715},{710,711},{712,713}};
  /* HF FED numbering of DCCs */
  int fedhfnum[EMAP_NHFCR][2]={{718,719},{720,721},{722,723}};
  /* HO FED numbering of DCCs */
  int fedhonum[EMAP_NHOCR][2]={{724,725},{726,727},{728,729},{730,731}};
  /* HBHE/HF htr slot offsets for set of three htrs */
  int ihslot[EMAP_NHSETS]={2,5,13,16};
  /* HO htr slot offsets for three sets of four htrs */
  int ihslotho[EMAP_NHSETSHO][EMAP_NHTRSHO]={{2,3,4,5},{6,7,13,14},{15,16,17,18}};
  /* iphi (lower) starting index for each HBHE crate */
  int ihbhephis[EMAP_NHBHECR]={11,19,3,27,67,35,59,43,51};
  /* iphi (lower) starting index for each HF crate */
  int ihfphis[EMAP_NHFCR]={3,27,51};
  /* iphi (lower) starting index for each HO crate */
  int ihophis[EMAP_NHOCR]={71,17,35,53};
  /* ihbheetadepth - unique HBHE {eta,depth} assignments per fiber and fiber channel */
  int ihbheetadepth[EMAP_NHTRS][EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) -bot(+top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}},
     {{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) +bot(-top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}}},
    {{{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) -bot(+top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}}},
     {{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) +bot(-top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}}}},
    {{{{28,1},{28,2},{29,1}},  /* htr 2 (HE) -bot(+top) */
      {{28,3},{24,2},{24,1}},
      {{27,1},{27,2},{29,2}},
      {{27,3},{23,2},{23,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}}},
     {{{27,1},{27,2},{29,2}},  /* htr 2 (HE) +bot(-top) */
      {{27,3},{23,2},{23,1}},
      {{28,1},{28,2},{29,1}},
      {{28,3},{24,2},{24,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}}}}
  };
  /* ihfetadepth - unique HF {eta,depth} assignments per fiber and fiber channel */
  int ihfetadepth[EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{33,1},{31,1},{29,1}},  /* top */
     {{32,1},{30,1},{34,1}},
     {{33,2},{31,2},{29,2}},
     {{32,2},{30,2},{34,2}},
     {{34,2},{32,2},{30,2}},
     {{31,2},{29,2},{33,2}},
     {{34,1},{32,1},{30,1}},
     {{31,1},{29,1},{33,1}}},
    {{{41,1},{37,1},{35,1}},  /* bot */
     {{38,1},{36,1},{39,1}},
     {{41,2},{37,2},{35,2}},
     {{38,2},{36,2},{39,2}},
     {{40,2},{38,2},{36,2}},
     {{37,2},{35,2},{39,2}},
     {{40,1},{38,1},{36,1}},
     {{37,1},{35,1},{39,1}}}
  };
  /* ihoetasidephi - unique HO {eta,side,phi} assignments per fiber and fiber channel */
  int ihoetasidephi[EMAP_NHTRSHO][EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][3]={
    {{{{ 1,-1,0},{ 2,-1,0},{ 3,-1,0}},  /* htr 0 (HO) top */
      {{ 1,-1,1},{ 2,-1,1},{ 3,-1,1}},
      {{ 1,-1,2},{ 2,-1,2},{ 3,-1,2}},
      {{ 1,-1,3},{ 2,-1,3},{ 3,-1,3}},
      {{ 1,-1,4},{ 2,-1,4},{ 3,-1,4}},
      {{ 1,-1,5},{ 2,-1,5},{ 3,-1,5}},
      {{14, 1,0},{14, 1,1},{14, 1,2}},
      {{14, 1,3},{14, 1,4},{14, 1,5}}},
     {{{ 1, 1,0},{ 2, 1,0},{ 3, 1,0}},  /* htr 0 (HO) bot */
      {{ 1, 1,1},{ 2, 1,1},{ 3, 1,1}},
      {{ 1, 1,2},{ 2, 1,2},{ 3, 1,2}},
      {{ 1, 1,3},{ 2, 1,3},{ 3, 1,3}},
      {{ 1, 1,4},{ 2, 1,4},{ 3, 1,4}},
      {{ 1, 1,5},{ 2, 1,5},{ 3, 1,5}},
      {{15, 1,0},{15, 1,1},{15, 1,2}},
      {{15, 1,3},{15, 1,4},{15, 1,5}}}},
    {{{{ 6, 1,0},{ 6, 1,1},{ 6, 1,2}},  /* htr 1 (HO) top */
      {{ 6, 1,3},{ 6, 1,4},{ 6, 1,5}},
      {{ 7, 1,0},{ 7, 1,1},{ 7, 1,2}},
      {{ 7, 1,3},{ 7, 1,4},{ 7, 1,5}},
      {{ 8, 1,0},{ 8, 1,1},{ 8, 1,2}},
      {{ 8, 1,3},{ 8, 1,4},{ 8, 1,5}},
      {{ 9, 1,0},{ 9, 1,1},{ 9, 1,2}},
      {{ 9, 1,3},{ 9, 1,4},{ 9, 1,5}}},
     {{{10, 1,0},{10, 1,1},{10, 1,2}},  /* htr 1 (HO) bot */
      {{10, 1,3},{10, 1,4},{10, 1,5}},
      {{11, 1,0},{11, 1,1},{11, 1,2}},
      {{11, 1,3},{11, 1,4},{11, 1,5}},
      {{12, 1,0},{12, 1,1},{12, 1,2}},
      {{12, 1,3},{12, 1,4},{12, 1,5}},
      {{13, 1,0},{13, 1,1},{13, 1,2}},
      {{13, 1,3},{13, 1,4},{13, 1,5}}}},
    {{{{ 4,-1,0},{ 4,-1,1},{ 0, 0,0}},  /* htr 2 (HO) top */
      {{ 4,-1,2},{ 4,-1,3},{ 0, 0,0}},
      {{ 4,-1,4},{ 4,-1,5},{ 0, 0,0}},
      {{ 0, 0,0},{ 0, 0,0},{ 0, 0,0}},
      {{ 5,-1,0},{ 5,-1,1},{ 5,-1,2}},
      {{ 5,-1,3},{ 5,-1,4},{ 5,-1,5}},
      {{14,-1,0},{14,-1,1},{14,-1,2}},
      {{14,-1,3},{14,-1,4},{14,-1,5}}},
     {{{ 4, 1,0},{ 4, 1,1},{ 0, 0,0}},  /* htr 2 (HO) bot */
      {{ 4, 1,2},{ 4, 1,3},{ 0, 0,0}},
      {{ 4, 1,4},{ 4, 1,5},{ 0, 0,0}},
      {{ 0, 0,0},{ 0, 0,0},{ 0, 0,0}},
      {{ 5, 1,0},{ 5, 1,1},{ 5, 1,2}},
      {{ 5, 1,3},{ 5, 1,4},{ 5, 1,5}},
      {{15,-1,0},{15,-1,1},{15,-1,2}},
      {{15,-1,3},{15,-1,4},{15,-1,5}}}},
    {{{{ 6,-1,0},{ 6,-1,1},{ 6,-1,2}},  /* htr 3 (HO) top */
      {{ 6,-1,3},{ 6,-1,4},{ 6,-1,5}},
      {{ 7,-1,0},{ 7,-1,1},{ 7,-1,2}},
      {{ 7,-1,3},{ 7,-1,4},{ 7,-1,5}},
      {{ 8,-1,0},{ 8,-1,1},{ 8,-1,2}},
      {{ 8,-1,3},{ 8,-1,4},{ 8,-1,5}},
      {{ 9,-1,0},{ 9,-1,1},{ 9,-1,2}},
      {{ 9,-1,3},{ 9,-1,4},{ 9,-1,5}}},
     {{{10,-1,0},{10,-1,1},{10,-1,2}},  /* htr 3 (HO) bot */
      {{10,-1,3},{10,-1,4},{10,-1,5}},
      {{11,-1,0},{11,-1,1},{11,-1,2}},
      {{11,-1,3},{11,-1,4},{11,-1,5}},
      {{12,-1,0},{12,-1,1},{12,-1,2}},
      {{12,-1,3},{12,-1,4},{12,-1,5}},
      {{13,-1,0},{13,-1,1},{13,-1,2}},
      {{13,-1,3},{13,-1,4},{13,-1,5}}}} 
  };
  int ic,is,ih,itb,ifb,ifc,ifwtb,iphi_loc;
  int iside,ieta,iphi,idepth,icrate,ihtr,ihtr_fi,ifi_ch,ispigot,idcc,ifed;
  //  int idcc_sl;
  std::string det;
  std::string fpga;
  // printf("      side       eta       phi     depth       det     crate       htr      fpga    htr_fi     fi_ch     spigo       dcc    dcc_sl     fedid\n");
  /* all HBHE crates */
  for(ic=0; ic<EMAP_NHBHECR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<EMAP_NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hbhecrate[ic];
	      iside=is<EMAP_NHSETS/2?-1:1;
	      ifwtb=(is/2+itb+1)%2;
	      ieta=ihbheetadepth[ih][ifwtb][ifb][ifc][0];
	      idepth=ihbheetadepth[ih][ifwtb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      det=((ieta>16||idepth>2)?("HE"):("HB"));
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>20)?(ihbhephis[ic]+(is%2)*4+itb*2-1)%72+1:(ihbhephis[ic]+(is%2)*4+itb*2+(ifb/2+is/2+1)%2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<EMAP_NHSETS/2?1:2;
	      //	      idcc_sl=idcc==1?9:19;
	      ifed=fedhbhenum[ic][idcc-1];
	      /// load map
	      HcalElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      HcalDetId hId((det=="HB")?(HcalBarrel):(HcalEndcap),ieta*iside,iphi,idepth);
	      emap.mapEId2chId(elId,hId);
	      
	      //	      printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  /* all HF crates */
  for(ic=0; ic<EMAP_NHFCR; ic++){
    /* four sets of three htrs per crate */
    for(is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for(ih=0; ih<EMAP_NHTRS; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hfcrate[ic];
	      iside=is<EMAP_NHSETS/2?-1:1;
	      ieta=ihfetadepth[itb][ifb][ifc][0];
	      idepth=ihfetadepth[itb][ifb][ifc][1];
	      ihtr=ihslot[is]+ih;
	      det="HF";
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ieta>39)?(ihfphis[ic]+(is%2)*12+ih*4-3)%72+1:(ihfphis[ic]+(is%2)*12+ih*4+(ifb/4)*2-1)%72+1;
	      ispigot=(is%2)*6+ih*2+itb;
	      idcc=is<EMAP_NHSETS/2?1:2;
	      //	      idcc_sl=idcc==1?9:19;
	      ifed=fedhfnum[ic][idcc-1];
	      HcalElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      HcalDetId hId(HcalForward,ieta*iside,iphi,idepth);
	      emap.mapEId2chId(elId,hId);
	      // printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  /* all HO crates */
  for(ic=0; ic<EMAP_NHOCR; ic++){
    /* three sets of four htrs per crate */
    for(is=0; is<EMAP_NHSETSHO; is++){
      /* four htrs per set */
      for(ih=0; ih<EMAP_NHTRSHO; ih++){
	/* top and bottom */
	for(itb=0; itb<EMAP_NTOPBOT; itb++){
	  /* eight fibers per HTR FPGA */
	  for(ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
	    for(ifc=0; ifc<EMAP_NFCH; ifc++){
	      icrate=hocrate[ic];
	      idepth=1;
	      ieta=ihoetasidephi[ih][itb][ifb][ifc][0];
	      iside=ihoetasidephi[ih][itb][ifb][ifc][1];
	      iphi_loc=ihoetasidephi[ih][itb][ifb][ifc][2];
	      ihtr=ihslotho[is][ih];
	      det="HO";
	      fpga=((itb%2)==1)?("bot"):("top");
	      ihtr_fi=ifb+1;
	      ifi_ch=ifc;
	      iphi=(ihophis[ic]+is*6+iphi_loc-1)%72+1;
	      ispigot=ihtr<9?(ihtr-2)*2+itb:(ihtr-13)*2+itb;
	      idcc=ihtr<9?1:2;
	      //	      idcc_sl=idcc==1?9:19;
	      ifed=fedhonum[ic][idcc-1];
	      HcalElectronicsId elId(ifi_ch, ihtr_fi, ispigot, ifed-700);
	      elId.setHTR(icrate, ihtr, (fpga=="top")?(1):(0));
	      if (ieta==0) { // unmapped 
		emap.mapEId2chId(elId,DetId(HcalDetId::Undefined));
	      } else {
		HcalDetId hId(HcalOuter,ieta*iside,iphi,idepth+3); // HO is officially "depth=4"
		emap.mapEId2chId(elId,hId);
	      }
	      // printf(" %9d %9d %9d %9d %9s %9d %9d %9s %9d %9d %9d %9d %9d %9d\n",iside,ieta,iphi,idepth,&det,icrate,ihtr,&fpga,ihtr_fi,ifi_ch,ispigot,idcc,idcc_sl,ifed);
	    }}}}}}
  

  emap.sort();

}

void HcalDbHardcode::makeHardcodeFrontEndMap(HcalFrontEndMap& emap) {

  std::stringstream mystream;
  std::string detector[6] = {"XX","HB","HE","HO","HF","HT"};
  /* ihbheetadepth - unique HBHE {eta,depth} assignments per fiber and fiber channel */
  int ihbheetadepth[EMAP_NHTRS][EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) -bot(+top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}},
     {{{11,1},{ 7,1},{ 3,1}},  /* htr 0 (HB) +bot(-top) */
      {{ 5,1},{ 1,1},{ 9,1}},
      {{11,1},{ 7,1},{ 3,1}},
      {{ 5,1},{ 1,1},{ 9,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}},
      {{10,1},{ 6,1},{ 2,1}},
      {{ 8,1},{ 4,1},{12,1}}}},
    {{{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) -bot(+top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}}},
     {{{16,2},{15,2},{14,1}},  /* htr 1 (HBHE) +bot(-top) */
      {{15,1},{13,1},{16,1}},
      {{16,2},{15,2},{14,1}},
      {{15,1},{13,1},{16,1}},
      {{17,1},{16,3},{25,1}},
      {{18,1},{18,2},{25,2}},
      {{17,1},{16,3},{26,1}},
      {{18,1},{18,2},{26,2}}}},
    {{{{28,1},{28,2},{29,1}},  /* htr 2 (HE) -bot(+top) */
      {{28,3},{24,2},{24,1}},
      {{27,1},{27,2},{29,2}},
      {{27,3},{23,2},{23,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}}},
     {{{27,1},{27,2},{29,2}},  /* htr 2 (HE) +bot(-top) */
      {{27,3},{23,2},{23,1}},
      {{28,1},{28,2},{29,1}},
      {{28,3},{24,2},{24,1}},
      {{19,2},{20,1},{21,2}},
      {{19,1},{20,2},{21,1}},
      {{19,2},{20,1},{22,2}},
      {{19,1},{20,2},{22,1}}}}
  };
  /* iphi (lower) starting index for each HBHE crate */
  int ihbhephis[EMAP_NHBHECR]={11,19,3,27,67,35,59,43,51};
  /* RM and the RM fiber for HB HE */
  int irm_rmfiHBHE[EMAP_NHTRS][EMAP_NTOPBOT][EMAP_NFBR][2]={
    {{{6,1},{7,1},{6,2},{7,2},{4,1},{5,1},{4,2},{5,2}},  // HTR 0 top
     {{6,3},{7,3},{6,4},{7,4},{4,3},{5,3},{4,4},{5,4}}}, // HTR 0 bot
    {{{2,1},{3,1},{2,2},{3,2},{2,1},{3,1},{2,2},{3,2}},  // HTR 1 top
     {{2,3},{3,3},{2,4},{3,4},{2,3},{3,3},{2,4},{3,4}}}, // HTR 1 bot
    {{{4,1},{5,1},{4,2},{5,2},{6,1},{7,1},{6,2},{7,2}},  // HTR 2 top
     {{4,3},{5,3},{4,4},{5,4},{6,3},{7,3},{6,4},{7,4}}}  // HTR 2 bot
  };

  /* all HBHE crates */
  for (int ic=0; ic<EMAP_NHBHECR; ic++){
    /* four sets of three htrs per crate */
    for (int is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for (int ih=0; ih<EMAP_NHTRS; ih++){
        /* top and bottom */
        for (int itb=0; itb<EMAP_NTOPBOT; itb++){
          /* eight fibers per HTR FPGA */
          for (int ifb=0; ifb<EMAP_NFBR; ifb++){
	    /* three channels per fiber */
  	    for (int ifc=0; ifc<EMAP_NFCH; ifc++){
              int iside  = is<EMAP_NHSETS/2 ? -1 : 1;
              int ifwtb  = (is/2+itb+1)%2;
              int ieta   = ihbheetadepth[ih][ifwtb][ifb][ifc][0];
              int idepth = ihbheetadepth[ih][ifwtb][ifb][ifc][1];
              HcalSubdetector subdet = (ieta>16||idepth>2) ? HcalEndcap : HcalBarrel;
	      std::string det = detector[subdet];
              int iphi   = (ieta>20) ?
		(ihbhephis[ic]+(is%2)*4+itb*2-1)%72+1 :
		(ihbhephis[ic]+(is%2)*4+itb*2+(ifb/2+is/2+1)%2-1)%72+1;
	      HcalDetId fId(subdet,iside*ieta,iphi,idepth);
	      char sidesign;
	      int irm(0), iwedge(0);
              if (iside == -1) {
                sidesign = 'M';
                irm    = irm_rmfiHBHE[ih][(itb + 1) % 2][ifb][1];
                if (ieta >= 21 && (irm == 1 || irm == 3)) 
		  iwedge = (iphi + 1 + irm + 1) / 4;
                else
		  iwedge = (iphi + irm + 1) / 4;
                if (iwedge > 18) iwedge -= 18;
              } else {
                sidesign = 'P';
                irm    = irm_rmfiHBHE[ih][itb][ifb][1];
                if (ieta >= 21 && (irm == 4 || irm == 2))
		  iwedge = (iphi + 1 - irm + 6) / 4;
                else
		  iwedge = (iphi - irm + 6) / 4;
                if (iwedge > 18) iwedge -= 18;
	      }
	      char tempbuff[30];
              sprintf (tempbuff,"%s%c%2.2i%c",det.c_str(),sidesign,iwedge,'\0');
	      mystream << tempbuff;
	      std::string rbx = mystream.str();
	      mystream.str("");
	      emap.loadObject(fId,irm,rbx);
	    }
	  }
	}
      }
    }
  } // End of HBHE

  /* ihfetadepth - unique HF {eta,depth} assignments per fiber and fiber channel */
  int ihfetadepth[EMAP_NTOPBOT][EMAP_NFBR][EMAP_NFCH][2]={
    {{{33,1},{31,1},{29,1}},  /* top */
     {{32,1},{30,1},{34,1}},
     {{33,2},{31,2},{29,2}},
     {{32,2},{30,2},{34,2}},
     {{34,2},{32,2},{30,2}},
     {{31,2},{29,2},{33,2}},
     {{34,1},{32,1},{30,1}},
     {{31,1},{29,1},{33,1}}},
    {{{41,1},{37,1},{35,1}},  /* bot */
     {{38,1},{36,1},{39,1}},
     {{41,2},{37,2},{35,2}},
     {{38,2},{36,2},{39,2}},
     {{40,2},{38,2},{36,2}},
     {{37,2},{35,2},{39,2}},
     {{40,1},{38,1},{36,1}},
     {{37,1},{35,1},{39,1}}}
  };
  /* iphi (lower) starting index for each HF crate */
  int ihfphis[EMAP_NHFCR]={3,27,51};
   /* all HF crates */
  for (int ic=0; ic<EMAP_NHFCR; ic++){
    /* four sets of three htrs per crate */
    for (int is=0; is<EMAP_NHSETS; is++){
      /* three htrs per set */
      for (int ih=0; ih<EMAP_NHTRS; ih++){
        /* top and bottom */
        for (int itb=0; itb<EMAP_NTOPBOT; itb++){
          /* eight fibers per HTR FPGA */
          for (int ifb=0; ifb<EMAP_NFBR; ifb++){
            /* three channels per fiber */
            for (int ifc=0; ifc<EMAP_NFCH; ifc++){
              int iside  = is<EMAP_NHSETS/2 ? -1 : 1;
              int ieta   = ihfetadepth[itb][ifb][ifc][0];
              int idepth = ihfetadepth[itb][ifb][ifc][1];
              HcalSubdetector subdet = HcalForward;
	      std::string det = detector[subdet];
              int iphi  = (ieta>39) ?
		(ihfphis[ic]+(is%2)*12+ih*4-1)%72+1 :
		(ihfphis[ic]+(is%2)*12+ih*4+(ifb/4)*2-1)%72+1;
	      char sidesign;
              if (iside == -1) {
                sidesign = 'M';
                if (ieta < 40){
                  if      (iphi == 1)     iphi = 71;
                  else if (iphi == 71)    iphi = 1;
                  else if (iphi % 4 == 1) iphi -= 2;
                  else if (iphi % 4 == 3) iphi += 2;
                  else                    edm::LogInfo("HcalCalib") << "Even iphi in HFM" << std::endl;
                }
              } else {
                sidesign = 'P';
              }
	      HcalDetId fId(subdet,iside*ieta,iphi,idepth);
	      int irm(0), hfphi(0);
              if ((iside == 1 && ieta == 40) || (iside == -1 && ieta == 41)){
                irm   = ((iphi + 1) / 2) % 36 + 1;
                hfphi = ((iphi + 1) / 6) % 12 + 1; 
              } else {
                irm   = ( iphi + 1) / 2;
                hfphi = (iphi - 1) / 6 + 1;
              }
              irm = (irm - 1) % 3 + 1;
	      char tempbuff[30];
              sprintf (tempbuff,"%s%c%2.2i%c",det.c_str(),sidesign,hfphi,'\0');
              mystream << tempbuff;
	      std::string rbx = mystream.str();
              mystream.str("");
	      emap.loadObject(fId,irm,rbx);
	    }
	  }
	}
      }
    }
  } // End of HF

  //RM for the HO as a function of eta, phi and side as implemented in complete_ho_map.txt
  //There are only 24 phi columns because after that it begins to repeat. The relevant variable is phi mod 24.
  //HX as the 16th eta entry
  int HO_RM_table[24][16][2] = {
    {{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
    {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{4,4},{4,4},{4,4},{4,4},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
    {{4,4},{4,4},{4,4},{4,4},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
    {{3,3},{3,3},{3,3},{3,3},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
    {{3,3},{3,3},{3,3},{3,3},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{2,2},{2,2},{2,2},{2,2},{4,4},{4,4},{4,4},{4,4},{4,4},{4,4},{2,4},{2,4},{2,4},{2,4},{2,4},{2,4}},
    {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{1,1},{1,1},{1,1},{1,1},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{2,2},{2,2},{2,2},{2,2},{3,3},{3,3},{3,3},{3,3},{3,3},{3,3},{1,3},{1,3},{1,3},{1,3},{1,3},{1,3}},
    {{2,2},{2,2},{2,2},{2,2},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{3,3},{3,3},{3,3},{3,3},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{3,1},{3,1},{3,1},{3,1},{3,1},{3,1}},
    {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}},
    {{1,1},{1,1},{1,1},{1,1},{2,2},{2,2},{2,2},{2,2},{2,2},{2,2},{4,2},{4,2},{4,2},{4,2},{4,2},{4,2}} };
  for (int iside = -1; iside < 2; iside += 2) {
    for (int iph = 0; iph < 72; iph++) {
      for (int iet = 0; iet < 15; iet++) {
	int iphi  = iph + 1;
        int ieta  = iet + 1;
	int depth = 4;
	HcalSubdetector subdet = HcalOuter;
	std::string det = detector[subdet];
        int sidear  = (iside == 1) ? 1 : 0;
	int phmod24 = iph % 24;
        int irm     = HO_RM_table[phmod24][iet][sidear];
        char sidesign = (iside == -1) ? 'M' : 'P';
	int ring(0), sector(0);
        if      (ieta <= 4)                ring = 0;
        else if (ieta >= 5 && ieta <= 10)  ring = 1;
        else                               ring = 2;
        for (int i = -2; i < iphi; i+=6) sector++;
        if (sector > 12) sector = 1; 
        if (ring != 0 && sector % 2 != 0) sector++;
	char tempbuff[30];
        if (ring == 0) 
	  sprintf (tempbuff,"%s%i%2.2d",det.c_str(),ring,sector);
        else 
	  sprintf (tempbuff,"%s%i%c%2.2d",det.c_str(),ring,sidesign,sector);
        mystream << tempbuff;
	std::string rbx = mystream.str();
        mystream.str("");
	HcalDetId fId(subdet,iside*ieta,iphi,depth);
	emap.loadObject(fId,irm,rbx);
      }
    }
  } // End of HO

  emap.sort();

}

