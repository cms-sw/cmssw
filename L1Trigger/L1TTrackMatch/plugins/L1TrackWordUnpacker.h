// Utilities used for unpacking the track word
//----------------------------------------------------------------------------
//  Authors:  Sweta Baradia, Suchandra Dutta, Subir Sarkar (February 2025)
//----------------------------------------------------------------------------

#ifndef L1Trigger_L1TTrackMatch_L1TrackWordUnpacker_HH
#define L1Trigger_L1TTrackMatch_L1TrackWordUnpacker_HH

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include <ap_fixed.h>
#include <ap_int.h>

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1trackunpacker {
  using L1TTTrackType = TTTrack<Ref_Phase2TrackerDigi_>;
  
  // For precision studies
  const unsigned int PT_INTPART_BITS{9}; 
  const unsigned int ETA_INTPART_BITS{3};
  const unsigned int kExtraGlobalPhiBit{4};

  using pt_intern     = ap_ufixed<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1, PT_INTPART_BITS, AP_TRN, AP_SAT>;
  using glbeta_intern = ap_fixed<TTTrack_TrackWord::TrackBitWidths::kTanlSize, ETA_INTPART_BITS, AP_TRN, AP_SAT>;
  using glbphi_intern = ap_int<TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit>;
  using z0_intern     = ap_int<TTTrack_TrackWord::TrackBitWidths::kZ0Size>;  // 40cm / 0.1
  using d0_intern     = ap_uint<TTTrack_TrackWord::TrackBitWidths::kD0Size>;
  
  inline  const unsigned int DoubleToBit(double value, unsigned int maxBits, double step) {
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
  inline const double FloatPtFromBits(const L1TTTrackType& track) {
    ap_uint<TTTrack_TrackWord::TrackBitWidths::kRinvSize - 1> ptBits = track.getRinvWord();
    pt_intern digipt;
    digipt.V = ptBits.range();
    return digipt.to_double();
  }
  inline const double FloatEtaFromBits(const L1TTTrackType& track) {
    TTTrack_TrackWord::tanl_t etaBits = track.getTanlWord();
    glbeta_intern digieta;
    digieta.V = etaBits.range();
    return digieta.to_double();
  }
  inline const double FloatPhiFromBits(const L1TTTrackType& track) {
    int sector = track.phiSector();
    double sector_phi_value = 0;
    if (sector < 5) {
      sector_phi_value = 2.0 * M_PI * sector / 9.0;
    } else {
      sector_phi_value = (-1.0 * M_PI + M_PI / 9.0 + (sector - 5) * 2.0 * M_PI / 9.0);
    }
    glbphi_intern trkphiSector = DoubleToBit(sector_phi_value, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
    glbphi_intern local_phiBits = 0;
    local_phiBits.V = track.getPhiWord();

    glbphi_intern local_phi =
      DoubleToBit(BitToDouble(local_phiBits, TTTrack_TrackWord::TrackBitWidths::kPhiSize, TTTrack_TrackWord::stepPhi0),
                  TTTrack_TrackWord::TrackBitWidths::kPhiSize + l1trackunpacker::kExtraGlobalPhiBit,
                  TTTrack_TrackWord::stepPhi0);
    glbphi_intern digiphi = local_phi + trkphiSector;
    return BitToDouble(digiphi, TTTrack_TrackWord::TrackBitWidths::kPhiSize + l1trackunpacker::kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
  }
  inline const double FloatZ0FromBits(const L1TTTrackType& track) {
    z0_intern trkZ = track.getZ0Word();
    return BitToDouble(trkZ, TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0);
  }
}  // namespace l1trackunpacker

#endif
