#ifndef L1TMUONPHASE2_PHASE2GMT_CONSTANTS
#define L1TMUONPHASE2_PHASE2GMT_CONSTANTS

#include "ap_int.h"

namespace Phase2L1GMT {

  // INPUT Dataformat
  // Track
  const int BITSTTCURV = 15;
  const int BITSTTCURV2 = 27;
  const int BITSTTPHI = 12;
  const int BITSTTTANL = 16;
  // Not used in the emulator now, but will in future
  const int BITSTTZ0 = 12;
  const int BITSTTD0 = 13;
  const int BITSTTCHI2 = 4;
  const int BITSTTBENDCHI2 = 3;
  const int BITSTTMASK = 7;
  const int BITSTTTRACKMVA = 3;
  const int BITSTTOTHERMVA = 6;

  // Bitwidth for common internal variables in GMT
  const int BITSPT = 13;
  const int BITSPHI = 13;
  const int BITSETA = 13;
  const int BITSZ0 = 10;
  const int BITSD0 = 12;

  //Muon ROI
  const int BITSSTUBCOORD = 8;
  const int BITSSTUBETA = 8;
  const int BITSSTUBID = 9;
  const int BITSSTUBPHIQUALITY = 4;
  const int BITSSTUBETAQUALITY = 4;
  const int BITSSTUBTIME = 8;
  const int BITSSTAMUONQUALITY = 6;
  const int BITSMUONBX = 4;

  //PreTrackMatherdMuon
  const int BITSMATCHQUALITY = 9;
  const int BITSMUONBETA = 4;

  //Track Muon Match
  const int BITSSIGMAETA = 4;
  const int BITSSIGMACOORD = 4;
  const int BITSPROPCOORD = 9;
  const int BITSPROPSIGMACOORD_A = 5;
  const int BITSPROPSIGMACOORD_B = 5;
  const int BITSPROPSIGMAETA_A = 5;
  const int BITSPROPSIGMAETA_B = 5;

  // OUTPUT Dataformat
  // Bitwidth for standalone muons to CL1 and GT
  const int BITSSAZ0 = 5;
  const int BITSSAD0 = 7;
  const int BITSSAQUALITY = 4;

  // Bitwidth for dataformat to GT
  const int BITSGTPT = 16;
  const int BITSGTPHI = 13;
  const int BITSGTETA = 14;
  const int BITSGTZ0 = 10;
  const int BITSGTD0 = 10;
  const int BITSGTQUALITY = 8;
  const int BITSGTISO = 4;

  const float maxCurv_ = 0.00855;  // 2 GeV pT Rinv is in cm
  const float maxPhi_ = 1.026;     // relative to the center of the sector
  const float maxTanl_ = 8.0;
  const float maxZ0_ = 30.;
  const float maxD0_ = 15.4;
  // Updated barrelLimit according to Karol, https://indico.cern.ch/event/1113802/#1-phase2-gmt-performance-and-i
  const int barrelLimit0_ = 1.4 / 0.00076699039 / 8;
  const int barrelLimit1_ = 1.1 / 0.00076699039 / 8;
  const int barrelLimit2_ = 0.95 / 0.00076699039 / 8;
  const int barrelLimit3_ = 0.95 / 0.00076699039 / 8;
  const int barrelLimit4_ = 0;

  // LSB
  const float LSBpt = 0.03125;
  const float LSBphi = 2. * M_PI / pow(2, BITSPHI);
  const float LSBeta = 2. * M_PI / pow(2, BITSETA);
  const float LSBGTz0 = 2. * maxZ0_ / pow(2, BITSZ0);
  const float LSBGTd0 = 2. * maxD0_ / pow(2, BITSD0);
  const float LSBSAz0 = 1.875;
  const float LSBSAd0 = 3.85;

  typedef ap_uint<64> wordtype;

  inline uint64_t twos_complement(long long int v, uint bits) {
    uint64_t mask = (1 << bits) - 1;
    if (v >= 0)
      return v & mask;
    else
      return (~(-v) + 1) & mask;
  }

  template <class T>
  inline int wordconcat(T& word, int bstart, long int input, int bitsize) {
    int bend = bstart + bitsize - 1;
    word.range(bend, bstart) = twos_complement(input, bitsize);
    return bend + 1;
  }

}  // namespace Phase2L1GMT
#endif
