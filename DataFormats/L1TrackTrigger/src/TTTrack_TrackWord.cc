////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author: Mike Hildreth
// date:   April 9, 2019
//
///////

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include <iostream>
#include <bitset>
#include <string>

//Constructor - turn track parameters into 96-bit word

TTTrack_TrackWord::TTTrack_TrackWord(const GlobalVector& Momentum,
                                     const GlobalPoint& POCA,
                                     double theRinv,
                                     double theChi2XY,
                                     double theChi2Z,
                                     double theBendChi2,
                                     unsigned int theHitPattern,
                                     unsigned int iSpare) {
  setTrackWord(Momentum, POCA, theRinv, theChi2XY, theChi2Z, theBendChi2, theHitPattern, iSpare);
}

TTTrack_TrackWord::TTTrack_TrackWord(unsigned int theRinv,
                                     unsigned int phi0,
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int theChi2XY,
                                     unsigned int theChi2Z,
                                     unsigned int theBendChi2,
                                     unsigned int theHitPattern,
                                     unsigned int iSpare)
    : iRinv(theRinv),
      iphi(phi0),
      itanl(tanl),
      iz0(z0),
      id0(d0),
      ichi2XY(theChi2XY),      //revert to other packing?  Or will be unpacked wrong
      ichi2Z(theChi2Z),        //revert to other packing?  Or will be unpacked wrong
      iBendChi2(theBendChi2),  // revert to ogher packing? Or will be unpacked wrong
      ispare(iSpare),
      iHitPattern(theHitPattern) {
  initialize();
}

// one for already-digitized values:

void TTTrack_TrackWord::setTrackWord(unsigned int theRinv,
                                     unsigned int phi0,
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int theChi2XY,
                                     unsigned int theChi2Z,
                                     unsigned int theBendChi2,
                                     unsigned int theHitPattern,
                                     unsigned int iSpare) {
  iRinv = theRinv;
  iphi = phi0;
  itanl = tanl;
  iz0 = z0;
  id0 = d0;
  ichi2XY = theChi2XY;      //revert to other packing?  Or will be unpacked wrong
  ichi2Z = theChi2Z;        //revert to other packing?  Or will be unpacked wrong
  iBendChi2 = theBendChi2;  // revert to ogher packing? Or will be unpacked wrong
  ispare = iSpare;
  iHitPattern = theHitPattern;

  initialize();
}

// one for floats:
void TTTrack_TrackWord::setTrackWord(const GlobalVector& Momentum,
                                     const GlobalPoint& POCA,
                                     double theRinv,
                                     double theChi2XY,
                                     double theChi2Z,
                                     double theBendChi2,
                                     unsigned int theHitPattern,
                                     unsigned int iSpare) {
  initialize();

  // first, derive quantities to be packed

  float rPhi = Momentum.phi();  // this needs to be phi relative to center of sector ****
  float rTanl = Momentum.z() / Momentum.perp();
  float rZ0 = POCA.z();
  float rD0 = POCA.perp();

  // bin, convert to integers, and pack

  unsigned int seg1, seg2, seg3, seg4;

  //tanl
  itanl = digitize_Signed(rTanl, NTanlBits, 0, valLSBTanl);

  //z0
  iz0 = digitize_Signed(rZ0, NZ0Bits, 0, valLSBZ0);

  //chi2 has non-linear bins
  ichi2XY = 0;
  for (unsigned int ibin = 0; ibin < Nchi2; ++ibin) {
    ichi2XY = ibin;
    if (theChi2XY < chi2Bins[ibin])
      break;
  }

  //chi2Z has non-linear bins
  ichi2Z = 0;
  for (unsigned int ibin = 0; ibin < Nchi2; ++ibin) {
    ichi2Z = ibin;
    if (theChi2Z < chi2ZBins[ibin])
      break;
  }

  //phi
  iphi = digitize_Signed(rPhi, NPhiBits, 0, valLSBPhi);

  //d0
  id0 = digitize_Signed(rD0, ND0Bits, 0, valLSBD0);

  //Rinv
  iRinv = digitize_Signed(theRinv, NCurvBits, 0, valLSBCurv);

  //bend chi2 - non-linear bins
  iBendChi2 = 0;
  for (unsigned int ibin = 0; ibin < NBchi2; ++ibin) {
    iBendChi2 = ibin;
    if (theBendChi2 < Bchi2Bins[ibin])
      break;
  }

  ispare = iSpare;

  // spare bits
  if (ispare > 0x3FFF)
    ispare = 0x3FFF;

  iHitPattern = theHitPattern;

  //set bits
  /*
    Current packing scheme. Any changes here ripple everywhere!
    
    uint word1 = 16 (tanl) + 12 (z0) + 4 (chi2) = 32 bits
    uint word2 = 12 (phi) + 13 (d0) + 7 (hitPattern) = 32 bits
    uint word3 = 15 (pT) + 3 (bend chi2) + 14 (spare/TMVA) = 32 bits
   */

  //now pack bits; leave hardcoded for now as am example of how this could work

  seg1 = (itanl << (nWordBits - (NTanlBits + 1)));          // extra bit or bits is for sign //16
  seg2 = (iz0 << (nWordBits - (NTanlBits + NZ0Bits + 2)));  //4
  seg3 = ichi2XY;

  //set bits

  TrackWord1 = seg1 + seg2 + seg3;

  //second 32-bit word

  seg1 = (iphi << (nWordBits - (NPhiBits + 1)));           //20
  seg2 = (id0 << (nWordBits - (NPhiBits + ND0Bits + 2)));  //7

  //HitMask
  seg3 = theHitPattern;

  //set bits

  TrackWord2 = seg1 + seg2 + seg3;

  //third 32-bit word

  seg1 = (iRinv << (nWordBits - (NCurvBits + 1)));                            //17
  seg2 = (iBendChi2 << (nWordBits - (NCurvBits + NBChi2Bits + 1)));           //14
  seg3 = (ichi2Z << (nWordBits - (NCurvBits + NBChi2Bits + NChi2Bits + 1)));  //10
  seg4 = ispare;

  TrackWord3 = seg1 + seg2 + seg3 + seg4;
}
// unpack

float TTTrack_TrackWord::unpack_itanl() {
  unsigned int bits = (TrackWord1 & maskTanL) >> (nWordBits - (NTanlBits + 1));  //16
  float unpTanl = unpack_Signed(bits, NTanlBits, valLSBTanl);
  return unpTanl;
}

float TTTrack_TrackWord::get_itanl() {
  float unpTanl = unpack_Signed(itanl, NTanlBits, valLSBTanl);
  return unpTanl;
}

unsigned int TTTrack_TrackWord::get_tanlBits() {
  //unsigned int bits =  (TrackWord1 & 0xFFFF0000) >> 16;
  return itanl;
}

float TTTrack_TrackWord::unpack_iz0() {
  unsigned int bits = (TrackWord1 & maskZ0) >> (nWordBits - (NTanlBits + NZ0Bits + 2));  //4
  float unpZ0 = unpack_Signed(bits, NZ0Bits, valLSBZ0);
  return unpZ0;
}

float TTTrack_TrackWord::get_iz0() {
  float unpZ0 = unpack_Signed(iz0, NZ0Bits, valLSBZ0);
  return unpZ0;
}

unsigned int TTTrack_TrackWord::get_z0Bits() {
  //unsigned int bits =   (TrackWord1 & 0x0000FFF0) >> 4;
  return iz0;
}

float TTTrack_TrackWord::unpack_ichi2XY() {
  unsigned int bits = (TrackWord1 & maskChi2XY);
  float unpChi2 = chi2Bins[bits];
  return unpChi2;
}

float TTTrack_TrackWord::get_ichi2XY() {
  float unpChi2 = chi2Bins[ichi2XY];
  return unpChi2;
}

unsigned int TTTrack_TrackWord::get_chi2XYBits() {
  //unsigned int bits = (TrackWord1 & 0x0000000F);
  return ichi2XY;
}

float TTTrack_TrackWord::unpack_iphi() {
  unsigned int bits = (TrackWord2 & maskPhi) >> (nWordBits - (NPhiBits + 1));  //20
  float unpPhi = unpack_Signed(bits, NPhiBits, valLSBPhi);
  return unpPhi;
}

float TTTrack_TrackWord::get_iphi() {
  float unpPhi = unpack_Signed(iphi, NPhiBits, valLSBPhi);
  return unpPhi;
}

unsigned int TTTrack_TrackWord::get_phiBits() {
  //unsigned int bits =   (TrackWord2 & 0xFFF00000) >> 20;
  return iphi;
}

float TTTrack_TrackWord::unpack_id0() {
  unsigned int bits = (TrackWord2 & maskD0) >> (nWordBits - (NPhiBits + ND0Bits + 2));  //7
  float unpD0 = unpack_Signed(bits, ND0Bits, valLSBD0);
  return unpD0;
}

float TTTrack_TrackWord::get_id0() {
  float unpD0 = unpack_Signed(id0, ND0Bits, valLSBD0);
  return unpD0;
}

unsigned int TTTrack_TrackWord::get_d0Bits() {
  //  unsigned int bits =   (TrackWord2 & 0x000FFF80) >> 7;
  return id0;
}

unsigned int TTTrack_TrackWord::unpack_hitPattern() {
  unsigned int bits = (TrackWord2 & maskHitPat);
  return bits;
}

unsigned int TTTrack_TrackWord::get_hitPattern() { return iHitPattern; }

float TTTrack_TrackWord::unpack_iRinv() {
  unsigned int bits = (TrackWord3 & maskRinv) >> (nWordBits - (NCurvBits + 1));  //17
  float unpCurv = unpack_Signed(bits, NCurvBits, valLSBCurv);
  return unpCurv;
}

float TTTrack_TrackWord::get_iRinv() {
  float unpCurv = unpack_Signed(iRinv, NCurvBits, valLSBCurv);
  return unpCurv;
}

float TTTrack_TrackWord::unpack_iBendChi2() {
  unsigned int bits = (TrackWord3 & maskBendChi2) >> (nWordBits - (NCurvBits + NBChi2Bits + 1));  //14
  float unpBChi2 = Bchi2Bins[bits];
  return unpBChi2;
}

float TTTrack_TrackWord::get_iBendChi2() {
  float unpBChi2 = Bchi2Bins[iBendChi2];
  return unpBChi2;
}

unsigned int TTTrack_TrackWord::get_BendChi2Bits() {
  unsigned int bits = (TrackWord3 & maskBendChi2) >> (nWordBits - (NCurvBits + NBChi2Bits + 1));  //14
  return bits;
}

float TTTrack_TrackWord::unpack_ichi2Z() {
  unsigned int bits = (TrackWord3 & maskChi2Z) >> (nWordBits - (NCurvBits + NBChi2Bits + NChi2Bits + 1));  //10
  float unpChi2Z = chi2ZBins[bits];
  return unpChi2Z;
}

float TTTrack_TrackWord::get_ichi2Z() {
  float unpChi2Z = chi2ZBins[ichi2Z];
  return unpChi2Z;
}

unsigned int TTTrack_TrackWord::unpack_ispare() {
  unsigned int bits = (TrackWord3 & maskSpare);
  return bits;
}

unsigned int TTTrack_TrackWord::get_ispare() { return ispare; }

unsigned int TTTrack_TrackWord::digitize_Signed(float var, unsigned int maxBit, unsigned int minBit, float lsb) {
  unsigned int nBits = (maxBit - minBit + 1);
  unsigned int myVar = std::floor(fabs(var) / lsb);
  unsigned int maxVal = (1 << (nBits - 1)) - 1;
  if (myVar > maxVal)
    myVar = maxVal;
  if (var < 0)
    myVar = (1 << nBits) - myVar;  // two's complement encoding
  unsigned int seg = myVar;
  return seg;
}

float TTTrack_TrackWord::unpack_Signed(unsigned int bits, unsigned int nBits, float lsb) {
  int isign = 1;
  unsigned int maxVal = (1 << nBits) - 1;
  if (bits & (1 << nBits)) {  //check sign
    isign = -1;
    bits = (1 << (nBits + 1)) - bits;  // if negative, flip everything for two's complement encoding
  }
  float unpacked = (float(bits & maxVal) + 0.5) * lsb;
  unpacked = isign * unpacked;
  return unpacked;
}

void TTTrack_TrackWord::initialize() {
  /* bits for packing, constants defined in TTTrack_TrackWord.h :

  signed quantities (one bit for sign):
  
  q/R = 14+1
  phi = 11+1  (relative to sector center)
  tanl = 15+1
  z0  = 11+1
  d0  = 12+1

  unsigned:

  chi2     = 4
  BendChi2 = 3
  hitPattern  = 7
  Spare    = 10
  chi2Z    = 4

  */

  // define bits, 1<<N = 2^N

  unsigned int CurvBins = (1 << NCurvBits);
  unsigned int phiBins = (1 << NPhiBits);
  unsigned int tanlBins = (1 << NTanlBits);
  unsigned int z0Bins = (1 << NZ0Bits);
  unsigned int d0Bins = (1 << ND0Bits);

  Nchi2 = (1 << NChi2Bits);
  NBchi2 = (1 << NBChi2Bits);

  valLSBCurv = maxCurv / float(CurvBins);
  valLSBPhi = maxPhi / float(phiBins);
  valLSBTanl = maxTanl / float(tanlBins);
  valLSBZ0 = maxZ0 / float(z0Bins);
  valLSBD0 = maxD0 / float(d0Bins);

  chi2Bins[0] = 0.25;
  chi2Bins[1] = 0.5;
  chi2Bins[2] = 1.0;
  chi2Bins[3] = 2.;
  chi2Bins[4] = 3.;
  chi2Bins[5] = 5.;
  chi2Bins[6] = 7.;
  chi2Bins[7] = 10.;
  chi2Bins[8] = 20.;
  chi2Bins[9] = 40.;
  chi2Bins[10] = 100.;
  chi2Bins[11] = 200.;
  chi2Bins[12] = 500.;
  chi2Bins[13] = 1000.;
  chi2Bins[14] = 3000.;

  chi2ZBins[0] = 0.25;
  chi2ZBins[1] = 0.5;
  chi2ZBins[2] = 1.0;
  chi2ZBins[3] = 2.;
  chi2ZBins[4] = 3.;
  chi2ZBins[5] = 5.;
  chi2ZBins[6] = 7.;
  chi2ZBins[7] = 10.;
  chi2ZBins[8] = 20.;
  chi2ZBins[9] = 40.;
  chi2ZBins[10] = 100.;
  chi2ZBins[11] = 200.;
  chi2ZBins[12] = 500.;
  chi2ZBins[13] = 1000.;
  chi2ZBins[14] = 3000.;

  Bchi2Bins[0] = 0.5;
  Bchi2Bins[1] = 1.25;
  Bchi2Bins[2] = 2.0;
  Bchi2Bins[3] = 3.0;
  Bchi2Bins[4] = 5.0;
  Bchi2Bins[5] = 10.;
  Bchi2Bins[6] = 50.;
};
