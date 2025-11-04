#include "L1Trigger/L1TTrackMatch/interface/TkTripletEmuAlgo.h"

namespace l1ttripletemu {

  std::vector<cos_lut_fixed_t> generateCosLUT() {
    float phi = 0;
    std::vector<cos_lut_fixed_t> cosLUT;
    double stepPhi = TTTrack_TrackWord::stepPhi0 * (1 << l1ttripletemu::kCosLUTShift);
    for (unsigned int LUT_idx = 0; LUT_idx < l1ttripletemu::kCosLUTBins; LUT_idx++) {
      cosLUT.push_back((cos_lut_fixed_t)(cos(phi)));
      phi += stepPhi;
    }
    return cosLUT;
  }

  std::vector<cosh_lut_fixed_t> generateCoshLUT() {
    float eta = 0;
    std::vector<cosh_lut_fixed_t> coshLUT;
    double stepEta = 2.5 / (1 << l1ttripletemu::kCoshLUTTableSize);
    for (unsigned int lut_idx = 0; lut_idx < l1ttripletemu::kCoshLUTBins; lut_idx++) {
      coshLUT.push_back((cosh_lut_fixed_t)(cosh(eta)));
      eta += stepEta;
    }
    return coshLUT;
  }

  std::vector<sinh_lut_fixed_t> generateSinhLUT() {
    float eta = 0;
    std::vector<sinh_lut_fixed_t> sinhLUT;
    double stepEta = 2.5 / (1 << l1ttripletemu::kSinhLUTTableSize);
    for (unsigned int lut_idx = 0; lut_idx < l1ttripletemu::kSinhLUTBins; lut_idx++) {
      sinhLUT.push_back((sinh_lut_fixed_t)(sinh(eta)));
      eta += stepEta;
    }
    return sinhLUT;
  }

  global_phi_t localToGlobalPhi(TTTrack_TrackWord::phi_t local_phi, global_phi_t sector_shift) {
    global_phi_t PhiMin = 0;
    global_phi_t PhiMax = 2 * M_PI / TTTrack_TrackWord::stepPhi0;

    // The initial word comes in as a uint; the correct bits, but not automatically using 2s compliment format.
    global_phi_t globalPhi = local_phi;

    // Once the word is in a larger, signed container, shift it down so that the negative numbers are automatically represented in 2s compliment.
    if (local_phi >= (1 << (TTTrack_TrackWord::TrackBitWidths::kPhiSize - 1)))
      globalPhi -= (1 << TTTrack_TrackWord::TrackBitWidths::kPhiSize);

    globalPhi += sector_shift;

    if (globalPhi < PhiMin) {
      globalPhi = globalPhi + PhiMax;
    } else if (globalPhi > PhiMax) {
      globalPhi = globalPhi - PhiMax;
    }

    return globalPhi;
  }

  std::vector<global_phi_t> generatePhiSliceLUT(unsigned int N) {
    float sliceCentre = 0.0;
    std::vector<global_phi_t> phiLUT;
    for (unsigned int q = 0; q <= N; q++) {
      phiLUT.push_back((global_phi_t)(sliceCentre / TTTrack_TrackWord::stepPhi0));
      sliceCentre += 2 * M_PI / N;
    }
    return phiLUT;
  }

}  // namespace l1ttripletemu
