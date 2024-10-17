#ifndef L1Trigger_L1TTrackMatch_L1TrackUnpacker_HH
#define L1Trigger_L1TTrackMatch_L1TrackUnpacker_HH
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1trackunpacker {
  //For precision studies
  const unsigned int PT_INTPART_BITS{9};
  const unsigned int ETA_INTPART_BITS{3};
  const unsigned int kExtraGlobalPhiBit{4};

  typedef ap_ufixed<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1, PT_INTPART_BITS, AP_RND_CONV, AP_SAT> pt_intern;
  typedef ap_fixed<TTTrack_TrackWord::TrackBitWidths::kTanlSize, ETA_INTPART_BITS, AP_TRN, AP_SAT> glbeta_intern;
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit> glbphi_intern;
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kZ0Size> z0_intern;  // 40cm / 0.1
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kD0Size> d0_intern;
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kRinvSize> rinv_intern;

  inline const unsigned int DoubleToBit(double value, unsigned int maxBits, double step) {
    unsigned int digitized_value = std::floor(std::abs(value) / step);
    unsigned int digitized_maximum = (1 << (maxBits - 1)) - 1;  // The remove 1 bit from nBits to account for the sign
    if (digitized_value > digitized_maximum)
      digitized_value = digitized_maximum;
    if (value < 0)
      digitized_value = (1 << maxBits) - digitized_value;  // two's complement encoding
    return digitized_value;
  }
  inline const double BitToDouble(unsigned int bits, unsigned int maxBits, double step) {
    int isign = 1;
    unsigned int digitized_maximum = (1 << maxBits) - 1;
    if (bits & (1 << (maxBits - 1))) {  // check the sign
      isign = -1;
      bits = (1 << (maxBits + 1)) - bits;
    }
    return (double(bits & digitized_maximum) + 0.5) * step * isign;
  }

}  // namespace l1trackunpacker
#endif
