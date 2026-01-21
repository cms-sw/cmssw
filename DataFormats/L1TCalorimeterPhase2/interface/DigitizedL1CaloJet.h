#ifndef DataFormats_L1TCalorimeterPhase2_DigitizedL1CaloJet_h
#define DataFormats_L1TCalorimeterPhase2_DigitizedL1CaloJet_h

#include <ap_int.h>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace l1tp2 {

  class DigitizedL1CaloJet {
  public:
    DigitizedL1CaloJet() { jetData = 0x0; }

    DigitizedL1CaloJet(ap_uint<64> data) { jetData = data; }

    //Constructor from digitized inputs
    DigitizedL1CaloJet(ap_uint<1> isValid,
                       ap_uint<16> pt,
                       ap_int<13> phi,
                       ap_int<14> eta) {
      jetData = ((ap_uint<64>)isValid) | (((ap_uint<64>)pt) << 1) | (((ap_int<64>)phi) << 17) |
                (((ap_uint<64>)eta) << 30);
    }

    // Constructor from float inputs
    DigitizedL1CaloJet(bool isValid_b,
                       float pt_f,
                       float phi_f,
                       float eta_f) {
      jetData = ((ap_uint<64>)digitizeIsValid(isValid_b)) | ((ap_uint<64>)digitizePt(pt_f) << 1) |
      ((ap_uint<64>)digitizePhi(phi_f) << 17) | ((ap_uint<64>)digitizeEta(eta_f) << 30);
    }

    ap_uint<64> data() const { return jetData; }

    // LSB getters
    float ptLSB() const { return LSB_PT; }
    float phiLSB() const { return LSB_PHI; }
    float etaLSB() const { return LSB_ETA; }

    // Data getters (digitized)
    ap_uint<1> isValid() const { return (jetData & 1); }
    ap_uint<16> pt() const { return ((jetData >> 1) & 0xFFFF); } // 16 1s = 0xFFFF
    ap_int<13> phi() const { return ((jetData >> 17) & 0x1FFF); } // 13 1s = 0x1FFF
    ap_int<14> eta() const { return ((jetData >> 30) & 0x3FFF); } // 14 1s = 0x3FFF

    // Data getters (floats/bools)
    bool isValidBool() const { return (bool)isValid(); }
    float ptFloat() const { return (pt() * ptLSB()); }
    float phiFloat() const { return (phi() * phiLSB()); }
    float etaFloat() const { return (eta() * etaLSB()); }

  private:
    // Digitized information as one value
    unsigned long long int jetData;
    
    // Constants
    static constexpr float LSB_PT = 0.03125;
    static constexpr float LSB_PHI = M_PI / 0x1000; // 2^12 = 0x1000
    static constexpr float LSB_ETA = M_PI / 0x1000;

    static constexpr unsigned int n_bits_pt = 16;
    static constexpr unsigned int n_bits_phi = 13;
    static constexpr unsigned int n_bits_eta = 14;

    // Private member functions for doing the digitization
    ap_uint<16> digitizePt(float pt_f) {
      float maxPt_f = (std::pow(2.0f, n_bits_pt) - 1) * LSB_PT;
      // If pT exceeds the maximum, saturate the value
      if (pt_f >= maxPt_f) {
        return (ap_uint<16>)0xFFFF; // 16 1s = 0b1111111111111111 = 0xFFFF
      }
      return (ap_uint<16>)(pt_f / LSB_PT);
    }

    ap_uint<13> digitizePhi(float phi_f) {
      float maxPhi_f = (std::pow(2.0f, n_bits_phi-1) - 1) * LSB_PHI;
      // If phi exceeds the maximum (very few values should), saturate the value
      if (phi_f >= maxPhi_f) {
        return (ap_uint<13>)0xFFF; // 12 1s in binary = 0xFFF (1 bit for sign)
      } else if (phi_f <= -maxPhi_f) {
        return (ap_uint<13>)-0xFFF;
      }
      return (ap_uint<13>)(phi_f / LSB_PHI);
    }

    ap_uint<14> digitizeEta(float eta_f) {
      float maxEta_f = (std::pow(2.0f, n_bits_eta-1) - 1) * LSB_ETA;
      // If eta exceeds the maximum, saturate the value
      if (eta_f >= maxEta_f) {
        return (ap_uint<14>)0x1FFF; // 13 1s in binary = 0x1FFF (1 bit for sign)
      } else if (eta_f <= -maxEta_f) {
        return (ap_uint<14>)-0x1FFF;
      }
      return (ap_uint<14>)(eta_f / LSB_ETA);
    }

    ap_uint<1> digitizeIsValid(bool isValid_b) { return (ap_uint<1>)isValid_b; }
  };

  // Concrete collection of output objects (with extra tuning information)
  typedef std::vector<l1tp2::DigitizedL1CaloJet> DigitizedL1CaloJetCollection;
}  // namespace l1tp2
#endif
