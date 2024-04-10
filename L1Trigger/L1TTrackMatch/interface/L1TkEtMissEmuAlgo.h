#ifndef L1Trigger_L1TTrackMatch_L1TkEtMissEmuAlgo_HH
#define L1Trigger_L1TTrackMatch_L1TkEtMissEmuAlgo_HH

#include <ap_int.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace that defines constants and types used by the EtMiss Emulation
// Includes functions for writing LUTs and converting to integer representations
namespace l1tmetemu {

  const unsigned int kInternalPtWidth{14};
  const unsigned int kPtMagSize{9};
  const unsigned int kMETSize{16};  // For output Magnitude default 16
  const unsigned int kMETMagSize{11};
  const unsigned int kMETPhiSize{13};  // For Output Phi default 13
  const unsigned int kEtExtra{4};
  const unsigned int kGlobalPhiExtra{4};
  const unsigned int kCosLUTSize{10};
  const unsigned int kCosLUTMagSize{1};
  const unsigned int kCosLUTTableSize{10};
  const unsigned int kCosLUTBins{(1 << kCosLUTTableSize) + 1};
  const unsigned int kCosLUTShift{TTTrack_TrackWord::TrackBitWidths::kPhiSize - kCosLUTTableSize};
  const unsigned int kAtanLUTSize{64};
  const unsigned int kAtanLUTMagSize{2};

  typedef ap_ufixed<kMETSize, kMETMagSize, AP_RND_CONV, AP_SAT> METWord_t;
  typedef ap_int<kMETPhiSize> METWordphi_t;
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kPhiSize + kGlobalPhiExtra> global_phi_t;
  typedef ap_ufixed<kCosLUTSize, kCosLUTMagSize, AP_RND_CONV, AP_SAT> cos_lut_fixed_t;
  typedef ap_ufixed<kAtanLUTSize, kAtanLUTMagSize, AP_RND_CONV, AP_SAT> atan_lut_fixed_t;
  typedef ap_fixed<kMETSize + kEtExtra, kMETMagSize + kEtExtra, AP_RND_CONV, AP_SAT> Et_t;
  typedef ap_fixed<kMETPhiSize + kEtExtra, 4, AP_RND_CONV, AP_SAT> metphi_fixed_t;
  typedef ap_ufixed<kMETPhiSize + kEtExtra + 7, kMETPhiSize - 2, AP_RND_CONV, AP_SAT> pi_bins_fixed_t;

  // Output definition as per interface document, only used when creating output format
  const double kMaxMET = 1 << kMETMagSize;  // 2 TeV
  const double kMaxMETPhi{2 * M_PI};

  const double kStepMETwordEt = kMaxMET / (1 << kMETSize);
  const double kStepMETwordPhi = kMaxMETPhi / (1 << kMETPhiSize);
  const double kBinsInPi = 1.0 / kStepMETwordPhi;

  // Enough symmetry in cos and sin between 0 and pi/2 to get all possible values
  // of cos and sin phi
  const double kMaxCosLUTPhi{M_PI / 2};

  const unsigned int kNSector{9};
  const unsigned int kNQuadrants{4};

  // Simple struct used for ouput of cordic
  struct EtMiss {
    METWord_t Et;
    METWordphi_t Phi;
  };

  std::vector<cos_lut_fixed_t> generateCosLUT();

  global_phi_t localToGlobalPhi(TTTrack_TrackWord::phi_t local_phi, global_phi_t sector_shift);

  std::vector<global_phi_t> generatePhiSliceLUT(unsigned int N);

  template <typename T>
  void printLUT(std::vector<T> lut, std::string module = "", std::string name = "") {
    edm::LogVerbatim log(module);
    log << "The " << name << "[" << lut.size() << "] values are ... \n" << std::setprecision(30);
    for (unsigned int i = 0; i < lut.size(); i++) {
      log << "\t" << i << "\t" << lut[i] << "\n";
    }
  }

}  // namespace l1tmetemu
#endif
