
#ifndef FIRMWARE_TkJetWord_h
#define FIRMWARE_TkJetWord_h

// Class to store the 128-bit TkJet word for L1 Track Trigger.
// Author: Emily MacDonald, updated by Benjamin Radburn-Smith (September 2022)
// 2nd update: George Karathanasis (Oct 2022)

#include <vector>
#include <ap_int.h>
#include <cassert>
#include <cmath>
#include <bitset>
#include <string>
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"

namespace l1t {

  class TkJetWord {
  public:
    // ----------constants, enums and typedefs ---------
    static constexpr double MAX_Z0 = 30.;
    static constexpr double MAX_ETA = 8.;

    enum TkJetBitWidths {
      kPtSize = 16,
      kPtMagSize = 11,
      kGlbEtaSize = 14,
      kGlbPhiSize = 13,
      kZ0Size = 10,
      kNtSize = 5,
      kXtSize = 4,
      kDispFlagSize = 1,
      kUnassignedSize = 65,
      kTkJetWordSize =
          kPtSize + kGlbEtaSize + kGlbPhiSize + kZ0Size + kNtSize + kXtSize + kDispFlagSize + kUnassignedSize,
    };

    enum TkJetBitLocations {
      kPtLSB = 0,
      kPtMSB = kPtLSB + TkJetBitWidths::kPtSize - 1,
      kGlbPhiLSB = kPtMSB + 1,
      kGlbPhiMSB = kGlbPhiLSB + TkJetBitWidths::kGlbPhiSize - 1,
      kGlbEtaLSB = kGlbPhiMSB + 1,
      kGlbEtaMSB = kGlbEtaLSB + TkJetBitWidths::kGlbEtaSize - 1,
      kZ0LSB = kGlbEtaMSB + 1,
      kZ0MSB = kZ0LSB + TkJetBitWidths::kZ0Size - 1,
      kNtLSB = kZ0MSB + 1,
      kNtMSB = kNtLSB + TkJetBitWidths::kNtSize - 1,
      kXtLSB = kNtMSB + 1,
      kXtMSB = kXtLSB + TkJetBitWidths::kXtSize - 1,
      kDispFlagLSB = kXtMSB + 1,
      kDispFlagMSB = kDispFlagLSB + TkJetBitWidths::kDispFlagSize - 1,
      kUnassignedLSB = kDispFlagMSB + 1,
      kUnassignedMSB = kUnassignedLSB + TkJetBitWidths::kUnassignedSize - 1,
    };

    typedef ap_ufixed<kPtSize, kPtMagSize, AP_TRN, AP_SAT> pt_t;
    typedef ap_int<kGlbEtaSize> glbeta_t;
    typedef ap_int<kGlbPhiSize> glbphi_t;
    typedef ap_int<kZ0Size> z0_t;   // 40cm / 0.1
    typedef ap_uint<kNtSize> nt_t;  //number of tracks
    typedef ap_uint<kXtSize> nx_t;  //number of tracks with xbit = 1
    typedef ap_uint<kDispFlagSize> dispflag_t;
    typedef ap_uint<TkJetBitWidths::kUnassignedSize> tkjetunassigned_t;  // Unassigned bits
    typedef std::bitset<TkJetBitWidths::kTkJetWordSize> tkjetword_bs_t;
    typedef ap_uint<TkJetBitWidths::kTkJetWordSize> tkjetword_t;

  public:
    // ----------Constructors --------------------------
    TkJetWord() {}
    TkJetWord(pt_t pt,
              glbeta_t eta,
              glbphi_t phi,
              z0_t z0,
              nt_t nt,
              nx_t nx,
              dispflag_t dispflag,
              tkjetunassigned_t unassigned);

    ~TkJetWord() {}

    // ----------copy constructor ----------------------
    TkJetWord(const TkJetWord& word) { tkJetWord_ = word.tkJetWord_; }

    // ----------operators -----------------------------
    TkJetWord& operator=(const TkJetWord& word) {
      tkJetWord_ = word.tkJetWord_;
      return *this;
    }

    // ----------member functions (getters) ------------
    // These functions return arbitarary precision words (lists of bits) for each quantity
    pt_t ptWord() const {
      pt_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kPtMSB, TkJetBitLocations::kPtLSB);
      return ret;
    }
    glbeta_t glbEtaWord() const {
      glbeta_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kGlbEtaMSB, TkJetBitLocations::kGlbEtaLSB);
      return ret;
    }
    glbphi_t glbPhiWord() const {
      glbphi_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kGlbPhiMSB, TkJetBitLocations::kGlbPhiLSB);
      return ret;
    }
    z0_t z0Word() const {
      z0_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kZ0MSB, TkJetBitLocations::kZ0LSB);
      return ret;
    }
    nt_t ntWord() const {
      nt_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kNtMSB, TkJetBitLocations::kNtLSB);
      return ret;
    }
    nx_t xtWord() const {
      nx_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kXtMSB, TkJetBitLocations::kXtLSB);
      return ret;
    }
    dispflag_t dispFlagWord() const {
      dispflag_t ret;
      ret.V = tkJetWord()(TkJetBitLocations::kDispFlagMSB, TkJetBitLocations::kDispFlagLSB);
      return ret;
    }

    tkjetunassigned_t unassignedWord() const {
      return tkJetWord()(TkJetBitLocations::kUnassignedMSB, TkJetBitLocations::kUnassignedLSB);
    }
    tkjetword_t tkJetWord() const { return tkjetword_t(tkJetWord_.to_string().c_str(), 2); }

    // These functions return the packed bits in integer format for each quantity
    // Signed quantities have the sign enconded in the left-most bit.
    unsigned int ptBits() const { return ptWord().to_uint(); }
    unsigned int glbEtaBits() const { return glbEtaWord().to_uint(); }
    unsigned int glbPhiBits() const { return glbPhiWord().to_uint(); }
    unsigned int z0Bits() const { return z0Word().to_uint(); }
    unsigned int ntBits() const { return ntWord().to_uint(); }
    unsigned int xtBits() const { return xtWord().to_uint(); }
    unsigned int dispFlagBits() const { return dispFlagWord().to_uint(); }
    unsigned int unassignedBits() const { return unassignedWord().to_uint(); }

    // These functions return the unpacked and converted values
    // These functions return real numbers converted from the digitized quantities by unpacking the 64-bit vertex word
    float pt() const { return ptWord().to_float(); }
    float glbeta() const {
      return unpackSignedValue(
          glbEtaWord(), TkJetBitWidths::kGlbEtaSize, (MAX_ETA) / (1 << TkJetBitWidths::kGlbEtaSize));
    }
    float glbphi() const {
      return unpackSignedValue(
          glbPhiWord(), TkJetBitWidths::kGlbPhiSize, (2. * std::abs(M_PI)) / (1 << TkJetBitWidths::kGlbPhiSize));
    }
    float z0() const {
      return unpackSignedValue(z0Word(), TkJetBitWidths::kZ0Size, MAX_Z0 / (1 << TkJetBitWidths::kZ0Size));
    }
    int nt() const { return (ap_ufixed<kNtSize + 2, kNtSize>(ntWord())).to_int(); }
    int xt() const { return (ap_ufixed<kXtSize + 2, kXtSize>(xtWord())).to_int(); }
    int dispflag() const { return (ap_ufixed<kDispFlagSize + 2, kDispFlagSize>(dispFlagWord())).to_int(); }
    unsigned int unassigned() const { return unassignedWord().to_uint(); }

    // ----------member functions (setters) ------------
    void setTkJetWord(pt_t pt,
                      glbeta_t eta,
                      glbphi_t phi,
                      z0_t z0,
                      nt_t nt,
                      nx_t nx,
                      dispflag_t dispflag,
                      tkjetunassigned_t unassigned);

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
    tkjetword_bs_t tkJetWord_;
  };

  typedef std::vector<l1t::TkJetWord> TkJetWordCollection;

}  // namespace l1t

#endif
