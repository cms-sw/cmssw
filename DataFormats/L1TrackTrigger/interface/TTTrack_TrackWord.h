////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author: Mike Hildreth
// date:   April 9, 2019
//
///////

#ifndef L1_TRACK_TRIGGER_TRACK_WORD_H
#define L1_TRACK_TRIGGER_TRACK_WORD_H

#include <iostream>
#include <string>
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TTTrack_TrackWord {
public:
  // Constructors:

  TTTrack_TrackWord() {}
  TTTrack_TrackWord(const GlobalVector& Momentum,
                    const GlobalPoint& POCA,
                    double Rinv,
                    double Chi2,  // would be xy chisq if chi2Z is non-zero
                    double Chi2Z,
                    double BendChi2,
                    unsigned int HitPattern,
                    unsigned int iSpare);

  TTTrack_TrackWord(unsigned int Rinv,
                    unsigned int phi0,
                    unsigned int tanl,
                    unsigned int z0,
                    unsigned int d0,
                    unsigned int Chi2XY,  // would be total chisq if chi2Z is zero
                    unsigned int Chi2Z,
                    unsigned int BendChi2,
                    unsigned int HitPattern,
                    unsigned int iSpare);

  void setTrackWord(const GlobalVector& Momentum,
                    const GlobalPoint& POCA,
                    double Rinv,
                    double Chi2XY,  // would be total chisq if chi2Z is zero
                    double Chi2Z,
                    double BendChi2,
                    unsigned int HitPattern,
                    unsigned int iSpare);

  void setTrackWord(unsigned int Rinv,
                    unsigned int phi0,
                    unsigned int tanl,
                    unsigned int z0,
                    unsigned int d0,
                    unsigned int Chi2XY,  // would be total chisq if chi2Z is zero
                    unsigned int Chi2Z,
                    unsigned int BendChi2,
                    unsigned int HitPattern,
                    unsigned int iSpare);

  // getters for unpacked and converted values.
  // These functions return real numbers converted from the digitized quantities using the LSB defined for each.

  float get_iRinv();
  float get_iphi();
  float get_itanl();
  float get_iz0();
  float get_id0();
  float get_ichi2XY();
  float get_ichi2Z();
  float get_iBendChi2();

  // separate functions for unpacking 96-bit track word

  float unpack_iRinv();
  float unpack_iphi();
  float unpack_itanl();
  float unpack_iz0();
  float unpack_id0();
  float unpack_ichi2XY();
  float unpack_ichi2Z();
  float unpack_iBendChi2();
  unsigned int unpack_ispare();
  unsigned int unpack_hitPattern();

  // getters for packed bits.
  // These functions return, literally, the packed bits in integer format for each quantity.
  // Signed quantities have the sign enconded in the left-most bit.

  unsigned int get_hitPattern();
  unsigned int get_ispare();  // will need a converter(s) when spare bits are defined

  unsigned int get_RinvBits();
  unsigned int get_phiBits();
  unsigned int get_tanlBits();
  unsigned int get_z0Bits();
  unsigned int get_d0Bits();
  unsigned int get_chi2XYBits();
  unsigned int get_chi2ZBits();
  unsigned int get_BendChi2Bits();

  // copy constructor

  TTTrack_TrackWord(const TTTrack_TrackWord& word) {
    initialize();

    iRinv = word.iRinv;
    iphi = word.iphi;
    itanl = word.itanl;
    iz0 = word.iz0;
    id0 = word.id0;
    ichi2XY = word.ichi2XY;
    ichi2Z = word.ichi2Z;
    iBendChi2 = word.iBendChi2;
    ispare = word.ispare;
    iHitPattern = word.iHitPattern;

    // three 32-bit packed words
    TrackWord1 = word.TrackWord1;
    TrackWord2 = word.TrackWord2;
    TrackWord3 = word.TrackWord3;
  }

  TTTrack_TrackWord& operator=(const TTTrack_TrackWord& word) {
    initialize();
    iRinv = word.iRinv;
    iphi = word.iphi;
    itanl = word.itanl;
    iz0 = word.iz0;
    id0 = word.id0;
    ichi2XY = word.ichi2XY;
    ichi2Z = word.ichi2Z;
    iBendChi2 = word.iBendChi2;
    ispare = word.ispare;
    iHitPattern = word.iHitPattern;

    // three 32-bit packed words
    TrackWord1 = word.TrackWord1;
    TrackWord2 = word.TrackWord2;
    TrackWord3 = word.TrackWord3;

    return *this;
  }

private:
  void initialize();

  unsigned int digitize_Signed(float var, unsigned int maxBit, unsigned int minBit, float lsb);

  float unpack_Signed(unsigned int bits, unsigned int nBits, float lsb);

  // individual data members (not packed into 96 bits)
  unsigned int iRinv;
  unsigned int iphi;
  unsigned int itanl;
  unsigned int iz0;
  unsigned int id0;
  unsigned int ichi2XY;
  unsigned int ichi2Z;
  unsigned int iBendChi2;
  unsigned int ispare;
  unsigned int iHitPattern;

  // three 32-bit packed words

  unsigned int TrackWord1;
  unsigned int TrackWord2;
  unsigned int TrackWord3;

  // values of least significant bit for digitization

  float valLSBCurv;
  float valLSBPhi;
  float valLSBTanl;
  float valLSBZ0;
  float valLSBD0;

  float chi2Bins[16];
  float chi2ZBins[16];
  float Bchi2Bins[8];

  unsigned int Nchi2;  // both chi2 have same packing
  unsigned int NBchi2;

  /* bits for packing: 
  signed quantities (one bit for sign): 

  q/R = 14+1 
  phi = 11+1  (relative to sector center)
  tanl = 15+1
  z0  = 11+1
  d0  = 12+1                                                                                                                                                      

  unsigned:
  chi2     = 4  (or chisqXY)
  BendChi2 = 3
  hitmask  = 7
  chi2Z    = 4
  Spare    = 10 

  */

  // signed quantities: total bits are these values plus one
  const unsigned int NCurvBits = 14;
  const unsigned int NPhiBits = 11;
  const unsigned int NTanlBits = 15;
  const unsigned int NZ0Bits = 11;
  const unsigned int ND0Bits = 12;

  // unsigned:
  const unsigned int NChi2Bits = 4;
  const unsigned int NBChi2Bits = 3;
  const unsigned int NHitsBits = 7;
  const unsigned int NSpareBits = 14;

  // establish binning
  const float maxCurv = 0.00855;  // 2 GeV pT Rinv is in cm
  const float maxPhi = 1.026;     // relative to the center of the sector
  const float maxTanl = 8.0;
  const float maxZ0 = 30.;
  const float maxD0 = 15.4;

  // List of offsets and masks.  In principle, all of these could be derived from first
  // principles given the number of bits. At some future time, we'll do this.

  const unsigned int maskRinv = 0xFFFE0000;
  const unsigned int maskPhi = 0xFFF00000;
  const unsigned int maskTanL = 0xFFFF0000;
  const unsigned int maskD0 = 0x000FFF80;
  const unsigned int maskZ0 = 0x0000FFF0;
  const unsigned int maskChi2XY = 0x0000000F;
  const unsigned int maskChi2Z = 0x00003C00;
  const unsigned int maskBendChi2 = 0x0001C000;
  const unsigned int maskHitPat = 0x0000007F;
  const unsigned int maskSpare = 0x000003FF;

  const unsigned int nWordBits = 32;

};  // end of class def

#endif
