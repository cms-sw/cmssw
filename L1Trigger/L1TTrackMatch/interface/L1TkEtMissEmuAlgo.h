#ifndef L1Trigger_L1TTrackMatch_L1TkEtMissEmuAlgo_HH
#define L1Trigger_L1TTrackMatch_L1TkEtMissEmuAlgo_HH

#include <ap_int.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Namespace that defines constants and types used by the EtMiss Emulation
// Includes functions for writing LUTs and converting to integer representations
namespace l1tmetemu {
  
  const unsigned int kInternalPtWidth{14};
  const unsigned int kPtMagSize{9};
  const unsigned int kEtExtra{4};
  const unsigned int kGlobalPhiExtra{3};
  const unsigned int kCosLUTSize{10};
  const unsigned int kCosLUTMagSize{1};

  typedef ap_uint<TTTrack_TrackWord::TrackBitWidths::kPhiSize + kGlobalPhiExtra> global_phi_t;
  typedef ap_ufixed<kCosLUTSize, kCosLUTMagSize, AP_RND_CONV, AP_SAT> cos_lut_fixed_t;
  typedef ap_fixed<kInternalPtWidth + kEtExtra + kEtExtra, kPtMagSize + kEtExtra, AP_RND_CONV, AP_SAT> Et_t;
  typedef ap_fixed<kInternalPtWidth + kPtMagSize + kEtExtra,2*kPtMagSize + kEtExtra, AP_RND_CONV, AP_SAT> E2t_t;

  // Output definition as per interface document, only used when creating output format
  const double kMaxMET{2048};  // 2 TeV
  const double kMaxMETPhi{2 * M_PI};

  const unsigned int kMETSize{16};     // For output Magnitude default 16
  const unsigned int kMETMagSize{11};
  const unsigned int kMETPhiSize{13};  // For Output Phi default 13

  typedef ap_ufixed<kMETSize,kMETMagSize> METWord_t;
  typedef ap_uint<kMETPhiSize> METWordphi_t;

  const double kStepMETwordEt =  kMaxMET / ( 1 << kMETSize);
  const double kStepMETwordPhi = kMaxMETPhi / ( 1 << kMETPhiSize);

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

  std::vector<cos_lut_fixed_t> generateCosLUT(unsigned int size);

  global_phi_t localToGlobalPhi(TTTrack_TrackWord::phi_t local_phi, global_phi_t sector_shift );
  
  std::vector<global_phi_t> generatePhiSliceLUT(unsigned int N);

  template <typename T>
  void printLUT(std::vector<T> lut, std::string module = "", std::string name = "") {
    edm::LogVerbatim log(module);
    log << "The " << name << "[" << lut.size() << "] values are ... \n";
    for (unsigned int i = 0; i < lut.size(); i++) {
      log << "\t" << i << "\t" << lut[i] << "\n";
    }
  }

}  // namespace l1tmetemu
#endif
