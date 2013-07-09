//
// F.Ratnikov (UMd), Dec 14, 2005
//
#include <vector>
#include <string> 

#include "CLHEP/Random/RandGauss.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"

HcalPedestal HcalDbHardcode::makePedestal (HcalGenericDetId fId, bool fSmear) {
  HcalPedestalWidth width = makePedestalWidth (fId);
  float value0 = fId.genericSubdet() == HcalGenericDetId::HcalGenForward ? 11. : 18.;  // fC
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) value0 = 10.;
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i) / 100.); // ignore correlations, assume 10K pedestal run 
      while (value [i] <= 0) value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i));
    }
  }
  HcalPedestal result (fId.rawId (), 
		       value[0], value[1], value[2], value[3]
		       );
  return result;
}


HcalPedestal HcalDbHardcode::makePedestal (HcalGenericDetId fId, bool fSmear, double lumi) {
  HcalPedestalWidth width = makePedestalWidth (fId, lumi);
  //  float value0 = fId.genericSubdet() == HcalGenericDetId::HcalGenForward ? 11. : 18.;  // fC

  // Temporary disabling of lumi-dependent pedestal to avoid it being too big
  // for TDC evaluations...
  //  float value0 = 4.* width.getWidth(0);  // to be far enough from 0
  float value0 = fId.genericSubdet() == HcalGenericDetId::HcalGenForward ? 11. : 18.;  // fC

  float value [4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i) / 100.); // ignore correlations, assume 10K pedestal run 
      while (value [i] <= 0) value [i] = CLHEP::RandGauss::shoot (value0, width.getWidth (i));
    }
  }
  HcalPedestal result (fId.rawId (), 
		       value[0], value[1], value[2], value[3]
		       );
  return result;
}

HcalPedestalWidth HcalDbHardcode::makePedestalWidth (HcalGenericDetId fId) {
  float value = 0;
  if      (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) value = 5.0;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) value = 5.0;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter)  value = 1.5;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward)value = 2.0;
  // everything in fC

  HcalPedestalWidth result (fId.rawId ());
  for (int i = 0; i < 4; i++) {
    double width = value;
    for (int j = 0; j < 4; j++) {
      result.setSigma (i, j, i == j ? width * width : 0);
    }
  } 
  return result;
}

// Upgrade option with lumi dependence, assuming factor ~20 for HB 
// while factor ~8 (~2.5 less) for HE at 3000 fb-1 
// Tab.1.6 (p.10) and Fig. 5.7 (p.91) of HCAL Upgrade TDR  

HcalPedestalWidth HcalDbHardcode::makePedestalWidth (HcalGenericDetId fId, double lumi) {
  float value = 0;
  double eff_lumi = lumi - 200.; // offset to account for actual putting of SiPMs into
                                 // operations
  if(eff_lumi < 0.) eff_lumi = 0.;
  if      (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) 
    value = 5.0 + 1.7 * sqrt(eff_lumi);
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) 
    value = 5.0 + 0.7 * sqrt(eff_lumi);
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter)  value = 1.5;
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward)value = 2.0;
  // everything in fC

  /*
  if (fId.isHcalDetId()) {
    HcalDetId cell = HcalDetId(fId);
    int sub    = cell.subdet();
    int dep    = cell.depth();
    int ieta   = cell.ieta();
    int iphi   = cell.iphi();
    
    std::cout << "HCAL subdet " << sub << "  (" << ieta << "," << iphi
	      << "," << dep << ") " << " noise = " << value << std::endl; 
  }
  */

  HcalPedestalWidth result (fId.rawId ());
  for (int i = 0; i < 4; i++) {
    double width = value;
    for (int j = 0; j < 4; j++) {
      result.setSigma (i, j, i == j ? width * width : 0);
    }
  } 
  return result;
}


HcalMCParam HcalDbHardcode::makeMCParam (HcalGenericDetId fId) {


  /*
  std::cout << std::endl << " HcalDbHardcode::makeMCParam   fId " 
	    << fId 
	    << "  fId.genericSubdet() = " << fId.genericSubdet() << std::endl;
  if(fId.isHcalZDCDetId()) {
    std::cout << "... ZDC " << std::endl;
 

    HcalZDCDetId cell(fId);
    int side   = cell.zside();
    int depth  = cell.depth();
    int ch     = cell.channel();
    std::cout << "ZDC  side/depth/chanel = " 
	      << side << "  " << depth << "  " << ch 
	      << std::endl;
  }
  else if (fId.isHcalDetId()) {
    HcalDetId cell = HcalDetId(fId);
    int sub    = cell.subdet();
    int dep    = cell.depth();
    int ieta   = cell.ieta();
    int iphi   = cell.iphi();
    
    std::cout << "  HCAL " 
	      << "  subdet " << sub << "  ieta " << ieta << "  iphi " << iphi
	      << "  depth " << dep << std::endl; 
  }
  else {  std::cout << "...Something else ! " << std::endl; }

  */

  int r1bit[5];
  int pulseShapeID = 125;  r1bit[0] = 9;     //  [0,9]
  int syncPhase    = 0;    r1bit[1] = 1;
  int binOfMaximum = 0;    r1bit[2] = 4;
  float phase      = -25.0;                  // [-25.0,25.0]
  float Xphase     = (phase + 32.0) * 4.0;   // never change this line 
                                             // (offset 50nsec,  0.25ns step)
  int Iphase = Xphase;     r1bit[3] = 8;     // [0,256] offset 50ns, .25ns step
  int timeSmearing = 0;    r1bit[4] = 1;     //  bool
  
 
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) { 

    syncPhase    = 1;                      // a0  bool
    binOfMaximum = 5;                      // a1
    phase        = 5.0;                    // a2  [-25.0,25.0]
    Xphase       = (phase + 32.0) * 4.0;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 1;                      // a3
    pulseShapeID = 201;                    // a4   201 - Zecotec shape
                                           //      202 - Hamamatsu

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) { 

    syncPhase    = 1;                      // a0  bool
    binOfMaximum = 5;                      // a1
    phase        = 5.0;                    // a2  [-25.0,25.0]
    Xphase       = (phase + 32.0) * 4.0;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 1;                      // a3
    pulseShapeID = 201;                    // a4

  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {

    syncPhase    = 1;                      // a0  bool
    binOfMaximum = 5;                      // a1
    phase        = 5.0;                    // a2  [-25.0,25.0]
    Xphase       = (phase + 32.0) * 4.0;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a3
    pulseShapeID = 201;                    // a4

    HcalDetId cell = HcalDetId(fId);
    if (cell.ieta() == 1 && cell.iphi() == 1) pulseShapeID = 125;

  }
  
  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) { 

    syncPhase    = 1;                      // a0  bool
    binOfMaximum = 3;                      // a1
    phase        = 14.0;                   // a2  [-25.0,25.0]
    Xphase       = (phase + 32.0) * 4.0;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a3
    pulseShapeID = 301;                    // a4
 
  }

  else if (fId.genericSubdet() == HcalGenericDetId::HcalGenZDC) { 

    //    std::cout << " >>>  ZDC  " << std::endl; 

    syncPhase    = 1;                      // a0  bool
    binOfMaximum = 5;                      // a1
    phase        = -4.0;                   // a2  [-25.0,25.0]
    Xphase       = (phase + 32.0) * 4.0;   // never change this line 
                                           // (offset 50nsec,  0.25ns step)
    Iphase       = Xphase;
    timeSmearing = 0;                      // a3
    pulseShapeID = 401;                    // a4
 
  }


  int rshift[7];
  rshift[0]=0;
  for(int k=0; k<5; k++) {
    int j=k+1;
    rshift[j]=rshift[k]+r1bit[k];
    //  cout<<"  j= "<<j<<"  shift "<< rshift[j]<<endl;
  }

  int packingScheme  = 1;
  unsigned int param = pulseShapeID |
    syncPhase<<rshift[1]            |
    (binOfMaximum << rshift[2])     |
    (Iphase << rshift[3])           |
    (timeSmearing << rshift[4] | packingScheme << 27);
  
  /*
    
  int a0 =  param%512;
  int a1 = (param/512)%2;
  int a2 = (param/(512*2))%16;
  int a3 = (param/(512*2*16))%256;
  int a4 = (param/(512*2*16*256))%2;
  a3 = (a3/4)-32;
  int a5 = (param/(512*2*16*256*2*16))%16;
  
  */
  

  // unpacking a la CondFormats/HcalObjects/interface/HcalMCParam.h

  /*  
  int shape         =  param&0x1FF;
  syncPhase     = (param>>9)&0x1;
  binOfMaximum  = (param>>10)&0xF;
  int timePhase     = ((param>>14)&0xFF)/4-32;
  timeSmearing  = (param>>22)&0x1;
  int packingSc     = (param>>27)&0xF;
       
  std::cout << "  shape " << shape << "  sync.phase " <<  syncPhase
	    << "  binOfMaximum " <<  binOfMaximum
	    << "  timePhase " << timePhase 
	    << "  timeSmear " << timeSmearing
	    << "  packingSc " << packingSc
	    << std::endl;
  */

  /*
  if(shape != a0) 
    { std::cout <<"  error   shape " << shape 
		<< "  a0 " << a0 << std::endl; }
  if(syncPhase != a1) 
    { std::cout << "  error   syncPhase " << syncPhase 
		<< "  a1 " << a1 << std::endl; }
  if(binOfMaximum != a2) 
    { std::cout << "  error   binOfMaximum " << binOfMaximum
		<< "  a2 " << a2 << std::endl; }
  if(timePhase !=  a3) 
    { std::cout << "  error   timePhase " << timePhase 
		<< "  a3 " << a3 << std::endl; }
  if(timeSmearing != a4) 
    { std::cout << "  error   timeSmearing " << timeSmearing 
		<< "  a4 " << a4 << std::endl; }
  if(packingSc != a5) 
    { std::cout << "  error   packing sceme " << packingSc 
		<< "  a5 " << a5 << std::endl; }
  */
 
  HcalMCParam result(fId.rawId(), param);
  return result;

}

HcalRecoParam HcalDbHardcode::makeRecoParam (HcalGenericDetId fId) {

  /*
  if (fId.isHcalDetId()) {
    HcalDetId cell = HcalDetId(fId);
    int sub    = cell.subdet();
    int dep    = cell.depth();
    int ieta   = cell.ieta();
    int iphi   = cell.iphi();
    
    std::cout << "  HCAL " 
	      << "  subdet " << sub << "  ieta " << ieta << "  iphi " << iphi
	      << "  depth " << dep << std::endl; 
  }
  */


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
  int pulseShapeID = 105;                  p1bit[5]=9;   // [0,9]

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
    pulseShapeID = 201;                    // p5
    
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
    pulseShapeID = 201;                    // p5
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
    pulseShapeID = 301;                    // p5

    firstSample  = 2;                      // p3
    samplesToAdd = 1;                      // p4
    pulseShapeID = 301;                    // p5
    
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
  
  // Test printout
  /*  
    int a0=param1%2;
    int a1=(param1/2)%2;
    int a2=(param1/(2*2))%256;
    int a3=(param1/(2*2*256))%16;
    int a4=(param1/(2*2*256*16))%16;
    int a5=(param1/(2*2*256*16*16))%512;
    a2=(a2/4)-32;
    
    int b0=param2%2;
    int b1=(param2/2)%16;
    int b2=(param2/(2*16))%2;
    int b3=(param2/(2*16*2))%16;
    int b4=(param2/(2*16*2*16))%2;
    int b5=(param2/(2*16*2*16*2))%16;
    int b6=(param2/(2*16*2*16*2*16))%16;
    int b7=(param2/(2*16*2*16*2*16*16))%16;
    int b8=(param2/(2*16*2*16*2*16*16*16))%16;
    int b9=(param2/(2*16*2*16*2*16*16*16*16))%16;

    std::cout << " param1 (a012) " << a0 << " " <<a1 << " " << a2 
	      << " (a345) " << a3 << " " << a4 << " "<< a5
	      << " param2 (b012) " << b0 << " " << b1 << " " <<b2 
	      << " (b345) " << b3 << " " << b4 << " " << b5 
	      << " (b678) " << b6 << " " << b7 << " " << b8  << "   b9 " << b9
	      << std::endl;
  */


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


HcalGain HcalDbHardcode::makeGain (HcalGenericDetId fId, bool fSmear) {
  HcalGainWidth width = makeGainWidth (fId);
  float value0 = 0;
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenBarrel) {
    if (HcalDetId(fId).depth() == 1) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 2) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 3) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 4) value0 = 0.003333;
    else value0 = 0.003333; // GeV/fC
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenEndcap) {
    if (HcalDetId(fId).depth() == 1) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 2) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 3) value0 = 0.003333;
    else if (HcalDetId(fId).depth() == 4) value0 = 0.003333;
    else value0 = 0.003333; // GeV/fC
    // if (fId.genericSubdet() != HcalGenericDetId::HcalGenForward) value0 = 0.177;  // GeV/fC
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    HcalDetId hid(fId);
    if ((hid.ieta() > -5) && (hid.ieta() < 5))
      value0 = 0.0125;
    else
      value0 = 0.02083;  // GeV/fC
  } else if (fId.genericSubdet() == HcalGenericDetId::HcalGenForward) {
    if (HcalDetId(fId).depth() == 1) value0 = 0.2146;
    else if (HcalDetId(fId).depth() == 2) value0 = 0.3375;
  } else value0 = 0.003333; // GeV/fC
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) for (int i = 0; i < 4; i++) value [i] = CLHEP::RandGauss::shoot (value0, width.getValue (i)); 
  HcalGain result (fId.rawId (), value[0], value[1], value[2], value[3]);
  return result;
}

HcalGainWidth HcalDbHardcode::makeGainWidth (HcalGenericDetId fId) {
  float value = 0;
  HcalGainWidth result (fId.rawId (), value, value, value, value);
  return result;
}

HcalQIECoder HcalDbHardcode::makeQIECoder (HcalGenericDetId fId) {
  HcalQIECoder result (fId.rawId ());
  float offset = 0.0;
  float slope = fId.genericSubdet () == HcalGenericDetId::HcalGenForward ? 
  0.36 : 0.333;  // ADC/fC

  // qie8/qie10 attribution - 0/1
  if (fId.genericSubdet() == HcalGenericDetId::HcalGenOuter) {
    result.setQIEIndex(0);
    slope = 1.0;
  } else 
    result.setQIEIndex(1);
    
  for (unsigned range = 0; range < 4; range++) {
    for (unsigned capid = 0; capid < 4; capid++) {
      result.setOffset (capid, range, offset);
      result.setSlope (capid, range, slope);
    }
  }

  return result;
}

HcalCalibrationQIECoder HcalDbHardcode::makeCalibrationQIECoder (HcalGenericDetId fId) {
  HcalCalibrationQIECoder result (fId.rawId ());
  float lowEdges [64];
  for (int i = 0; i < 64; i++) lowEdges[i] = -1.5 + i*1.0;
  result.setMinCharges (lowEdges);
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape () {

  //  std::cout << " !!! HcalDbHardcode::makeQIEShape " << std::endl; 

  return HcalQIEShape ();
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
