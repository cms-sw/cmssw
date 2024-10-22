#ifndef L1Trigger_L1TTrackMatch_L1TkHTMissEmulatorProducer_HH
#define L1Trigger_L1TTrackMatch_L1TkHTMissEmulatorProducer_HH

// Original Author:  Hardik Routray
//         Created:  Mon, 11 Oct 2021

#include <ap_int.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"

// Namespace that defines constants and types used by the HTMiss Emulation

namespace l1tmhtemu {

  const unsigned int kInternalPtWidth{l1t::TkJetWord::TkJetBitWidths::kPtSize};
  const unsigned int kInternalEtaWidth{l1t::TkJetWord::TkJetBitWidths::kGlbEtaSize};
  const unsigned int kInternalPhiWidth{l1t::TkJetWord::TkJetBitWidths::kGlbPhiSize};

  // extra room for sumPx, sumPy
  const unsigned int kEtExtra{10};
  const unsigned int kValidSize{1};
  const unsigned int kMHTSize{16};  // For output Magnitude default 16
  const unsigned int kMHTIntSize{11};
  const unsigned int kMHTPhiSize{13};  // For output Phi default 13
  const unsigned int kHTSize{kInternalPtWidth + kEtExtra};
  const unsigned int kUnassignedSize{64 - (kHTSize + kMHTSize + kMHTPhiSize + kValidSize)};

  enum BitLocations {
    // The location of the least significant bit (LSB) and most significant bit (MSB) in the sum word for different fields
    kValidLSB = 0,
    kValidMSB = kValidLSB + kValidSize - 1,
    kMHTLSB = kValidMSB + 1,
    kMHTMSB = kMHTLSB + kMHTSize - 1,
    kMHTPhiLSB = kMHTMSB + 1,
    kMHTPhiMSB = kMHTPhiLSB + kMHTPhiSize - 1,
    kHTLSB = kMHTPhiMSB + 1,
    kHTMSB = kHTLSB + kHTSize - 1,
    kUnassignedLSB = kHTMSB + 1,
    kUnassignedMSB = kUnassignedLSB + kUnassignedSize - 1
  };

  const float kMaxMHT{2048};  // 2 TeV
  const float kMaxMHTPhi{2 * M_PI};

  typedef ap_uint<5> ntracks_t;
  typedef ap_uint<kInternalPtWidth> pt_t;
  typedef ap_int<kInternalEtaWidth> eta_t;
  typedef ap_int<kInternalPhiWidth> phi_t;

  typedef ap_int<kHTSize> Et_t;
  typedef ap_ufixed<kMHTSize, kMHTIntSize> MHT_t;
  typedef ap_uint<kMHTPhiSize> MHTphi_t;

  const unsigned int kMHTBins = 1 << kMHTSize;
  const unsigned int kMHTPhiBins = 1 << kMHTPhiSize;

  const double kStepPt{0.25};
  const double kStepEta{M_PI / (720)};
  const double kStepPhi{M_PI / (720)};

  const double kStepMHT = (l1tmhtemu::kMaxMHT / l1tmhtemu::kMHTBins);
  const double kStepMHTPhi = (l1tmhtemu::kMaxMHTPhi / l1tmhtemu::kMHTPhiBins);

  const unsigned int kPhiBins = 1 << kInternalPhiWidth;

  const float kMaxCosLUTPhi{M_PI};

  template <typename T>
  T digitizeSignedValue(double value, unsigned int nBits, double lsb) {
    T digitized_value = std::floor(std::abs(value) / lsb);
    T digitized_maximum = (1 << (nBits - 1)) - 1;  // The remove 1 bit from nBits to account for the sign
    if (digitized_value > digitized_maximum)
      digitized_value = digitized_maximum;
    if (value < 0)
      digitized_value = (1 << nBits) - digitized_value;  // two's complement encoding
    return digitized_value;
  }

  inline std::vector<phi_t> generateCosLUT(unsigned int size) {  // Fill cosine LUT with integer values
    float phi = 0;
    std::vector<phi_t> cosLUT;
    for (unsigned int LUT_idx = 0; LUT_idx < size; LUT_idx++) {
      cosLUT.push_back(digitizeSignedValue<phi_t>(cos(phi), l1tmhtemu::kInternalPhiWidth, l1tmhtemu::kStepPhi));
      phi += l1tmhtemu::kStepPhi;
    }
    cosLUT.push_back((phi_t)(0));  //Prevent overflow in last bin
    return cosLUT;
  }

  inline std::vector<MHTphi_t> generateaTanLUT(int cordicSteps) {  // Fill atan LUT with integer values
    std::vector<MHTphi_t> atanLUT;
    atanLUT.reserve(cordicSteps);
    for (int cordicStep = 0; cordicStep < cordicSteps; cordicStep++) {
      atanLUT.push_back(MHTphi_t(round((kMHTPhiBins * atan(pow(2, -1 * cordicStep))) / (2 * M_PI))));
    }
    return atanLUT;
  }

  inline std::vector<Et_t> generatemagNormalisationLUT(int cordicSteps) {
    float val = 1.0;
    std::vector<Et_t> magNormalisationLUT;
    for (int cordicStep = 0; cordicStep < cordicSteps; cordicStep++) {
      val = val / (pow(1 + pow(4, -1 * cordicStep), 0.5));
      magNormalisationLUT.push_back(Et_t(round(kMHTBins * val)));
    }
    return magNormalisationLUT;
  }

  struct EtMiss {
    MHT_t Et;
    MHTphi_t Phi;
  };

  inline EtMiss cordicSqrt(Et_t x,
                           Et_t y,
                           int cordicSteps,
                           std::vector<l1tmhtemu::MHTphi_t> atanLUT,
                           std::vector<Et_t> magNormalisationLUT) {
    Et_t new_x = 0;
    Et_t new_y = 0;

    MHTphi_t phi = 0;
    MHTphi_t new_phi = 0;
    bool sign = false;

    EtMiss ret_etmiss;

    if (x >= 0 && y >= 0) {
      phi = 0;
      sign = true;
      //x = x;
      //y = y;
    } else if (x < 0 && y >= 0) {
      phi = kMHTPhiBins >> 1;
      sign = false;
      x = -x;
      //y = y;
    } else if (x < 0 && y < 0) {
      phi = kMHTPhiBins >> 1;
      sign = true;
      x = -x;
      y = -y;
    } else {
      phi = kMHTPhiBins;
      sign = false;
      //x = x;
      y = -y;
    }

    for (int step = 0; step < cordicSteps; step++) {
      if (y < 0) {
        new_x = x - (y >> step);
        new_y = y + (x >> step);
      } else {
        new_x = x + (y >> step);
        new_y = y - (x >> step);
      }

      if ((y < 0) == sign) {
        new_phi = phi - atanLUT[step];
      } else {
        new_phi = phi + atanLUT[step];
      }

      x = new_x;
      y = new_y;
      phi = new_phi;
    }

    float sqrtval = (float(x * magNormalisationLUT[cordicSteps - 1]) / (float)kMHTBins) * float(kStepPt * kStepPhi);
    ret_etmiss.Et = sqrtval;
    ret_etmiss.Phi = phi;

    return ret_etmiss;
  }

}  // namespace l1tmhtemu
#endif
