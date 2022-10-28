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

  const unsigned int kInternalVTXWidth{12};  //default 8   max 12
  const unsigned int kInternalEtaWidth{8};   //default 8 max 16
  const unsigned int kInternalPtWidth{15};   // default 15  max 15
  const unsigned int kInternalPhiWidth{8};   //default 8  max  12

  // Extra bits needed by global phi to span full range
  const unsigned int kGlobalPhiExtra{3};  // default 3
  // Extra room for Et sums
  const unsigned int kEtExtra{10};  // default 8

  typedef ap_uint<3> nstub_t;

  typedef ap_uint<kInternalPhiWidth + kGlobalPhiExtra> global_phi_t;

  typedef ap_uint<kInternalPtWidth> pt_t;
  typedef ap_uint<kInternalEtaWidth> eta_t;
  typedef ap_uint<kInternalVTXWidth> z_t;
  // For internal Et representation, sums become larger than initial pt
  // representation
  typedef ap_int<kInternalPtWidth + kEtExtra> Et_t;

  //Output format
  const float kMaxMET{4096};  // 4 TeV
  const float kMaxMETPhi{2 * M_PI};
  const unsigned int kMETSize{15};     // For output Magnitude default 15
  const unsigned int kMETPhiSize{14};  // For Output Phi default 14

  typedef ap_uint<kMETSize> MET_t;
  // Cordic means this is evaluated between 0 and 2Pi rather than -pi to pi so
  // unsigned
  typedef ap_uint<kMETPhiSize> METphi_t;

  const unsigned int kGlobalPhiBins = 1 << kInternalPhiWidth;
  const unsigned int kMETBins = 1 << kMETSize;
  const unsigned int kMETPhiBins = 1 << kMETPhiSize;

  const unsigned int kNEtaRegion{6};
  const unsigned int kNSector{9};
  const unsigned int kNQuadrants{4};

  const float kMaxTrackZ0{-TTTrack_TrackWord::minZ0};
  const float kMaxTrackPt{512};
  const float kMaxTrackEta{4};

  // Steps used to convert from track word floats to track MET integer representations

  const double kStepPt = (std::abs(kMaxTrackPt)) / (1 << kInternalPtWidth);
  const double kStepEta = (2 * std::abs(kMaxTrackEta)) / (1 << kInternalEtaWidth);
  const double kStepZ0 = (2 * std::abs(kMaxTrackZ0)) / (1 << kInternalVTXWidth);

  const double kStepPhi = (2 * -TTTrack_TrackWord::minPhi0) / (kGlobalPhiBins - 1);

  const double kStepMET = (l1tmetemu::kMaxMET / l1tmetemu::kMETBins);
  const double kStepMETPhi = (l1tmetemu::kMaxMETPhi / l1tmetemu::kMETPhiBins);

  // Enough symmetry in cos and sin between 0 and pi/2 to get all possible values
  // of cos and sin phi
  const float kMaxCosLUTPhi{M_PI / 2};

  // Simple struct used for ouput of cordic
  struct EtMiss {
    MET_t Et;
    METphi_t Phi;
  };

  std::vector<global_phi_t> generateCosLUT(unsigned int size);
  std::vector<eta_t> generateEtaRegionLUT(std::vector<double> EtaRegions);
  std::vector<z_t> generateDeltaZLUT(std::vector<double> DeltaZBins);

  template <typename T>
  T digitizeSignedValue(double value, unsigned int nBits, double lsb) {
    // Digitize the incoming value
    int digitizedValue = std::floor(value / lsb);

    // Calculate the maxmum possible positive value given an output of nBits in size
    int digitizedMaximum = (1 << (nBits - 1)) - 1;  // The remove 1 bit from nBits to account for the sign
    int digitizedMinimum = -1. * (digitizedMaximum + 1);

    // Saturate the digitized value
    digitizedValue = std::clamp(digitizedValue, digitizedMinimum, digitizedMaximum);

    // Do the two's compliment encoding
    T twosValue = digitizedValue;
    if (digitizedValue < 0) {
      twosValue += (1 << nBits);
    }

    return twosValue;
  }

  template <typename T>
  unsigned int getBin(double value, const T& bins) {
    auto up = std::upper_bound(bins.begin(), bins.end(), value);
    return (up - bins.begin() - 1);
  }

  int unpackSignedValue(unsigned int bits, unsigned int nBits);

  unsigned int transformSignedValue(unsigned int bits, unsigned int oldnBits, unsigned int newnBits);

}  // namespace l1tmetemu
#endif
