#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"

using namespace std;

namespace l1tmetemu {
  std::vector<cos_lut_fixed_t> generateCosLUT(unsigned int size) {  // Fill cosine LUT with integer values
    float phi = 0;
    std::vector<cos_lut_fixed_t> cosLUT;
    for (unsigned int LUT_idx = 0; LUT_idx < size; LUT_idx++) {
      cosLUT.push_back((cos_lut_fixed_t)(cos(phi)));
      phi += TTTrack_TrackWord::stepPhi0;
      //std::cout << LUT_idx << "," << (cos_lut_fixed_t)(cos(phi)) << std::endl;
    }
    cosLUT.push_back((cos_lut_fixed_t)(0));  //Prevent overflow in last bin
    return cosLUT;
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

}  // namespace l1tmetemu
