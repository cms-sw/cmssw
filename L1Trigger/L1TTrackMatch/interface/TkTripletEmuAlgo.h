#ifndef L1Trigger_L1TTrackmatch_TkTripletEmuAlgo_HH
#define L1Trigger_L1TTrackmatch_TkTripletEmuAlgo_HH

#include <ap_int.h>
#include <ap_fixed.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/TkTripletWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// Namespace that defines constants and types used by the TkTriplet collection

namespace l1ttripletemu {

  // Triplet word definitions
  const unsigned int kValidSize{1};
  const unsigned int kPtSize{16};
  const unsigned int kPtMagSize{11};
  const unsigned int kPhiSize{13};
  const unsigned int kEtaSize{14};
  const unsigned int kMassSize{13};
  const unsigned int kMassMagSize{7};
  const unsigned int kTrk1PtSize{13};
  const unsigned int kTrk1PtMagSize{8};
  const unsigned int kTrk2PtSize{13};
  const unsigned int kTrk2PtMagSize{8};
  const unsigned int kTrk3PtSize{13};
  const unsigned int kTrk3PtMagSize{8};
  const unsigned int kChargeSize{1};
  const unsigned int kUnassignedSize{128 - (kChargeSize + kTrk3PtSize + kTrk2PtSize + kTrk1PtSize + kMassSize +
                                            kEtaSize + kPhiSize + kPtSize + kValidSize)};

  // Track properties
  const unsigned int kTkPtSize{14};
  const unsigned int kTkPtMagSize{9};
  const unsigned int kTkEtaSize{16};
  const unsigned int kTkEtaMagSize{3};

  // LUT definitions (cos, cosh, sinh)
  const unsigned int kGlobalPhiExtra{4};
  const unsigned int kCosLUTSize{10};
  const unsigned int kCoshLUTSize{10};
  const unsigned int kSinhLUTSize{10};
  const unsigned int kCosLUTMagSize{1};
  const unsigned int kCoshLUTMagSize{3};
  const unsigned int kSinhLUTMagSize{3};
  const unsigned int kCosLUTTableSize{10};
  const unsigned int kCoshLUTTableSize{10};
  const unsigned int kSinhLUTTableSize{10};
  const unsigned int kCosLUTBins{(1 << kCosLUTTableSize) + 1};    // size of cos LUT
  const unsigned int kCoshLUTBins{(1 << kCoshLUTTableSize) + 1};  // size of cosh LUT
  const unsigned int kSinhLUTBins{(1 << kSinhLUTTableSize) + 1};  // size of sinh LUT
  const unsigned int kCosLUTShift{TTTrack_TrackWord::TrackBitWidths::kPhiSize - kCosLUTTableSize};

  // Triplet word type definitions
  typedef ap_uint<kValidSize> tktriplet_valid_t;
  typedef ap_ufixed<kPtSize, kPtMagSize, AP_RND_CONV, AP_SAT> tktriplet_pt_t;
  typedef ap_int<kPhiSize> tktriplet_phi_t;
  typedef ap_int<kEtaSize> tktriplet_eta_t;
  typedef ap_ufixed<kMassSize, kMassMagSize, AP_RND_CONV, AP_SAT> tktriplet_mass_t;
  typedef ap_ufixed<kTrk1PtSize, kTrk1PtMagSize, AP_RND_CONV, AP_SAT> tktriplet_trk_pt_t;
  typedef ap_uint<kChargeSize> tktriplet_charge_t;
  typedef ap_uint<kUnassignedSize> tktriplet_unassigned_t;

  typedef ap_ufixed<kMassSize + 7, kMassMagSize + 7, AP_RND_CONV, AP_SAT> tktriplet_mass_sq_t;

  typedef ap_ufixed<kTkPtSize, kTkPtMagSize, AP_RND_CONV, AP_SAT> pt_t;
  typedef ap_fixed<kTkPtSize + 1, kTkPtMagSize + 1, AP_RND_CONV, AP_SAT> pxyz_t;
  typedef ap_fixed<kTkEtaSize, kTkEtaMagSize, AP_RND_CONV, AP_SAT> eta_t;
  typedef ap_int<TTTrack_TrackWord::TrackBitWidths::kPhiSize + kGlobalPhiExtra> global_phi_t;
  typedef ap_int<kCosLUTTableSize + 1> cos_lut_index_t;
  typedef ap_ufixed<kCosLUTSize, kCosLUTMagSize, AP_RND_CONV, AP_SAT> cos_lut_fixed_t;
  typedef ap_int<kCoshLUTTableSize + 1> cosh_lut_index_t;
  typedef ap_ufixed<kCoshLUTSize, kCoshLUTMagSize, AP_RND_CONV, AP_SAT> cosh_lut_fixed_t;
  typedef ap_int<kSinhLUTTableSize + 1> sinh_lut_index_t;
  typedef ap_ufixed<kSinhLUTSize, kSinhLUTMagSize, AP_RND_CONV, AP_SAT> sinh_lut_fixed_t;

  // Output definition as per interface document, only used when creating output format
  const double kMaxMass = 1 << kMassMagSize;
  const double kStepMass = kMaxMass / (1 << kMassSize);

  const unsigned int kNSector{9};
  const unsigned int kNQuadrants{4};

  // Track triplet object
  struct TkTriplet {
    tktriplet_valid_t valid;
    tktriplet_pt_t pt;
    tktriplet_phi_t phi;
    tktriplet_eta_t eta;
    tktriplet_mass_t mass;
    tktriplet_trk_pt_t trk1Pt;
    tktriplet_trk_pt_t trk2Pt;
    tktriplet_trk_pt_t trk3Pt;
    tktriplet_charge_t charge;
    tktriplet_unassigned_t unassigned;
  };

  // ---- FUNCTION DEFINITIONS ----
  // Generate LUTs for W invariant mass calculation
  std::vector<cos_lut_fixed_t> generateCosLUT();
  std::vector<cosh_lut_fixed_t> generateCoshLUT();
  std::vector<sinh_lut_fixed_t> generateSinhLUT();

  // Local to global phi conversion util
  global_phi_t localToGlobalPhi(TTTrack_TrackWord::phi_t local_phi, global_phi_t sector_shift);
  std::vector<global_phi_t> generatePhiSliceLUT(unsigned int N);

  // Print LUT for debugging
  template <typename T>
  void printLUT(std::vector<T> lut, std::string module = "", std::string name = "") {
    edm::LogVerbatim log(module);
    log << "The " << name << "[" << lut.size() << "] values are ... \n" << std::setprecision(30);
    for (unsigned int i = 0; i < lut.size(); i++) {
      log << "\t" << i << "\t" << lut[i] << "\n";
    }
  }

}  // namespace l1ttripletemu
#endif
