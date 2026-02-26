#ifndef FIRMWARE_TkTripletWord_h
#define FIRMWARE_TkTripletWord_h

// Class to store the 128-bit TkTriplet word for the L1TrackTriggerMatch.
// Original author: George Karathanasis (Dec 2023)

#include <vector>
#include <ap_int.h>
#include <cassert>
#include <cmath>
#include <bitset>
#include <string>
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"

namespace l1t {

  class TkTripletWord {
  public:
    // ----------constants, enums and typedefs ---------
    static constexpr double MAX_MASS = 1000.;
    static constexpr double MAX_ETA = 8.;
    static constexpr double MAX_CHARGE = 3.;
    static constexpr double MAX_Z0 = 25.;

    enum TkTripletBitWidths {
      kValidSize = 1,        // Width of the valid bit
      kPtSize = 16,          // Width of the triplet pT
      kPtMagSize = 11,       // Width of the triplet pT magnitude
      kPhiSize = 13,         // Width of the triplet phi
      kEtaSize = 14,         // Width of the triplet eta
      kMassSize = 13,        // Width of the triplet invariant mass
      kMassMagSize = 7,      // Width of the triplet invariant mass magnitude
      kTrk1PtSize = 13,      // Width of the leading triplet track pT
      kTrk1PtMagSize = 8,    // Width of the leading triplet track pT magnitude
      kTrk2PtSize = 13,      // Width of the subleading triplet track pT
      kTrk2PtMagSize = 8,    // Width of the subleading triplet track pT magnitude
      kTrk3PtSize = 13,      // Width of the third triplet track pT
      kTrk3PtMagSize = 8,    // Width of the third triplet track pT magnitude
      kChargeSize = 1,       // Width of the triplet charge (positive or negative)
      kUnassignedSize = 31,  // Width of unassigned bits

      // Output word size is a sum of the individual fields
      kTkTripletWordSize = kValidSize + kPtSize + kPhiSize + kEtaSize + kMassSize + kTrk1PtSize + kTrk2PtSize +
                           kTrk3PtSize + kChargeSize + kUnassignedSize,
    };

    enum TkTripletBitLocations {
      // The location of the least significant bit (LSB) and most significant bit (MSB) in the triplet word for different fields
      kValidLSB = 0,
      kValidMSB = kValidLSB + TkTripletBitWidths::kValidSize - 1,
      kPtLSB = kValidMSB + 1,
      kPtMSB = kPtLSB + TkTripletBitWidths::kPtSize - 1,
      kPhiLSB = kPtMSB + 1,
      kPhiMSB = kPhiLSB + TkTripletBitWidths::kPhiSize - 1,
      kEtaLSB = kPhiMSB + 1,
      kEtaMSB = kEtaLSB + TkTripletBitWidths::kEtaSize - 1,
      kMassLSB = kEtaMSB + 1,
      kMassMSB = kMassLSB + TkTripletBitWidths::kMassSize - 1,
      kTrk1PtLSB = kMassMSB + 1,
      kTrk1PtMSB = kTrk1PtLSB + TkTripletBitWidths::kTrk1PtSize - 1,
      kTrk2PtLSB = kTrk1PtMSB + 1,
      kTrk2PtMSB = kTrk2PtLSB + TkTripletBitWidths::kTrk2PtSize - 1,
      kTrk3PtLSB = kTrk2PtMSB + 1,
      kTrk3PtMSB = kTrk3PtLSB + TkTripletBitWidths::kTrk3PtSize - 1,
      kChargeLSB = kTrk3PtMSB + 1,
      kChargeMSB = kChargeLSB + TkTripletBitWidths::kChargeSize - 1,
      kUnassignedLSB = kChargeMSB + 1,
      kUnassignedMSB = kUnassignedLSB + TkTripletBitWidths::kUnassignedSize - 1,
    };

    // ap parameter types
    typedef ap_uint<TkTripletBitWidths::kValidSize> tktriplet_valid_t;
    typedef ap_ufixed<TkTripletBitWidths::kPtSize, TkTripletBitWidths::kPtMagSize, AP_RND_CONV, AP_SAT> tktriplet_pt_t;
    typedef ap_int<TkTripletBitWidths::kPhiSize> tktriplet_phi_t;
    typedef ap_int<TkTripletBitWidths::kEtaSize> tktriplet_eta_t;
    typedef ap_ufixed<TkTripletBitWidths::kMassSize, TkTripletBitWidths::kMassMagSize, AP_RND_CONV, AP_SAT>
        tktriplet_mass_t;
    typedef ap_ufixed<TkTripletBitWidths::kMassSize + 7, TkTripletBitWidths::kMassMagSize + 7, AP_RND_CONV, AP_SAT>
        tktriplet_mass_sq_t;
    typedef ap_ufixed<TkTripletBitWidths::kTrk1PtSize, TkTripletBitWidths::kTrk1PtMagSize, AP_RND_CONV, AP_SAT>
        tktriplet_trk_pt_t;
    typedef ap_uint<TkTripletBitWidths::kChargeSize> tktriplet_charge_t;
    typedef ap_uint<TkTripletBitWidths::kUnassignedSize> tktriplet_unassigned_t;

    typedef std::bitset<TkTripletBitWidths::kTkTripletWordSize> tktripletword_bs_t;
    typedef ap_uint<TkTripletBitWidths::kTkTripletWordSize> tktripletword_t;

  public:
    // ----------Constructors --------------------------
    TkTripletWord() {}
    TkTripletWord(tktriplet_valid_t valid,
                  tktriplet_pt_t pt,
                  tktriplet_phi_t phi,
                  tktriplet_eta_t eta,
                  tktriplet_mass_t mass,
                  tktriplet_trk_pt_t trk1Pt,
                  tktriplet_trk_pt_t trk2Pt,
                  tktriplet_trk_pt_t trk3Pt,
                  tktriplet_charge_t charge,
                  tktriplet_unassigned_t unassigned);

    ~TkTripletWord() {}

    // ----------copy constructor ----------------------
    TkTripletWord(const TkTripletWord& word) { tkTripletWord_ = word.tkTripletWord_; }

    // ----------operators -----------------------------
    TkTripletWord& operator=(const TkTripletWord& word) {
      tkTripletWord_ = word.tkTripletWord_;
      return *this;
    }

    // ----------member functions (getters) ------------
    // These functions return arbitarary precision words (lists of bits) for each quantity
    tktriplet_valid_t validWord() const {
      return tkTripletWord()(TkTripletBitLocations::kValidMSB, TkTripletBitLocations::kValidLSB);
    }

    tktriplet_pt_t ptWord() const {
      tktriplet_pt_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kPtMSB, TkTripletBitLocations::kPtLSB);
      return ret;
    }

    tktriplet_phi_t phiWord() const {
      tktriplet_phi_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kPhiMSB, TkTripletBitLocations::kPhiLSB);
      return ret;
    }

    tktriplet_eta_t etaWord() const {
      tktriplet_eta_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kEtaMSB, TkTripletBitLocations::kEtaLSB);
      return ret;
    }

    tktriplet_mass_t massWord() const {
      tktriplet_mass_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kMassMSB, TkTripletBitLocations::kMassLSB);
      return ret;
    }

    tktriplet_trk_pt_t trk1PtWord() const {
      tktriplet_trk_pt_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kTrk1PtMSB, TkTripletBitLocations::kTrk1PtLSB);
      return ret;
    }

    tktriplet_trk_pt_t trk2PtWord() const {
      tktriplet_trk_pt_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kTrk2PtMSB, TkTripletBitLocations::kTrk2PtLSB);
      return ret;
    }

    tktriplet_trk_pt_t trk3PtWord() const {
      tktriplet_trk_pt_t ret;
      ret.V = tkTripletWord()(TkTripletBitLocations::kTrk3PtMSB, TkTripletBitLocations::kTrk3PtLSB);
      return ret;
    }

    tktriplet_charge_t chargeWord() const {
      return tkTripletWord()(TkTripletBitLocations::kChargeMSB, TkTripletBitLocations::kChargeLSB);
    }

    tktriplet_unassigned_t unassignedWord() const {
      return tkTripletWord()(TkTripletBitLocations::kUnassignedMSB, TkTripletBitLocations::kUnassignedLSB);
    }

    tktripletword_t tkTripletWord() const { return tktripletword_t(tkTripletWord_.to_string().c_str(), 2); }

    // These functions return the packed bits in integer format for each quantity
    // Signed quantities have the sign enconded in the left-most bit.
    unsigned int validBits() const { return validWord().to_uint(); }
    unsigned int massBits() const { return massWord().to_uint(); }
    unsigned int unassignedBits() const { return unassignedWord().to_uint(); }

    // These functions return the unpacked and converted values
    // These functions return real numbers converted from the digitized quantities by unpacking the 64-bit vertex word
    bool valid() const { return validWord().to_bool(); }
    float mass() const {
      return unpackSignedValue(
          massWord(), TkTripletBitWidths::kMassSize, MAX_MASS / (1 << TkTripletBitWidths::kMassSize));
    }
    unsigned int unassigned() const { return unassignedWord().to_uint(); }

    // ----------member functions (setters) ------------
    void setTkTripletWord(tktriplet_valid_t valid,
                          tktriplet_pt_t pt,
                          tktriplet_phi_t phi,
                          tktriplet_eta_t eta,
                          tktriplet_mass_t mass,
                          tktriplet_trk_pt_t trk1Pt,
                          tktriplet_trk_pt_t trk2Pt,
                          tktriplet_trk_pt_t trk3Pt,
                          tktriplet_charge_t charge,
                          tktriplet_unassigned_t unassigned);

    template <class packVarType>
    inline void packIntoWord(unsigned int& currentOffset, unsigned int wordChunkSize, packVarType& packVar);

  private:
    // ----------private member functions --------------
    double unpackSignedValue(unsigned int bits, unsigned int nBits, double lsb) const {
      int isign = 1;
      unsigned int digitized_maximum = (1 << nBits) - 1;
      if (bits & (1 << (nBits - 1))) {  // check the sign
        isign = -1;
        bits = (1 << (nBits + 1)) - bits;  // if negative, flip everything for two's complement encoding
      }
      return (double(bits & digitized_maximum) + 0.5) * lsb * isign;
    }

    // ----------member data ---------------------------
    tktripletword_bs_t tkTripletWord_;
  };

  typedef std::vector<l1t::TkTripletWord> TkTripletWordCollection;

}  // namespace l1t

#endif
