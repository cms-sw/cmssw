#ifndef FIRMWARE_TkLightMesonWord_h
#define FIRMWARE_TkLightMesonWord_h
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
//
//-----------------------------------------------------------------------------
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (January 2025)
//----------------------------------------------------------------------------- 
#include <vector>
#include <ap_int.h>
#include <cassert>
#include <cmath>
#include <bitset>
#include <string>

namespace l1t {

  class TkLightMesonWord {
  public:
    // ----------constants, enums and typedefs ---------
    int INTPHI_PI = 720;
    int INTPHI_TWOPI = 2 * INTPHI_PI;
    float INTPT_LSB = 1 >> 5;
    static constexpr double ETAPHI_LSB = M_PI / (1 << 12);
    static constexpr double Z0_LSB = 0.05;

    // change to typed enum
    enum TkLightMesonTypes {
      kPhiType = 1,
      kRhoType = 2,
      kBsType = 3,
    };

    enum TkLightMesonBitWidths {
      kValidSize = 1,
      kPtSize = 16,
      kPtMagSize = 11,
      kGlbPhiSize = 13,
      kGlbEtaSize = 14,
      kZ0Size = 10,
      kMassSize = 12,
      kMassMagSize = 3,
      kTypeSize = 2,
      kNtracksSize = 3,
      kIndexSize = 8,
      kUnassignedSize = 9,
      kTkLightMesonWordSize = 
        kValidSize + kPtSize + kGlbPhiSize + kGlbEtaSize + kZ0Size + kMassSize + kTypeSize + 
        kNtracksSize + 2 * kIndexSize + kUnassignedSize
    };

    enum TkLightMesonBitLocations {
      kValidLSB = 0,
      kValidMSB = kValidLSB + TkLightMesonBitWidths::kValidSize - 1,
      kPtLSB = kValidMSB + 1,
      kPtMSB = kPtLSB + TkLightMesonBitWidths::kPtSize - 1,
      kGlbPhiLSB = kPtMSB + 1,
      kGlbPhiMSB = kGlbPhiLSB + TkLightMesonBitWidths::kGlbPhiSize - 1,
      kGlbEtaLSB = kGlbPhiMSB + 1,
      kGlbEtaMSB = kGlbEtaLSB + TkLightMesonBitWidths::kGlbEtaSize - 1,
      kZ0LSB = kGlbEtaMSB + 1,
      kZ0MSB = kZ0LSB + TkLightMesonBitWidths::kZ0Size - 1,
      kMassLSB = kZ0MSB + 1,
      kMassMSB = kMassLSB + TkLightMesonBitWidths::kMassSize - 1,
      kTypeLSB = kMassMSB + 1,
      kTypeMSB = kTypeLSB + TkLightMesonBitWidths::kTypeSize - 1,
      kNtracksLSB = kTypeMSB + 1,
      kNtracksMSB = kNtracksLSB + TkLightMesonBitWidths::kNtracksSize - 1,
      kFirstIndexLSB = kNtracksMSB + 1,
      kFirstIndexMSB = kFirstIndexLSB + TkLightMesonBitWidths::kIndexSize - 1,
      kSecondIndexLSB = kFirstIndexMSB + 1,
      kSecondIndexMSB = kSecondIndexLSB + TkLightMesonBitWidths::kIndexSize - 1,
      kUnassignedLSB = kSecondIndexMSB + 1,
      kUnassignedMSB = kUnassignedLSB + TkLightMesonBitWidths::kUnassignedSize - 1
    };

    using valid_t      = ap_uint<TkLightMesonBitWidths::kValidSize>;
    using pt_t         = ap_ufixed<TkLightMesonBitWidths::kPtSize, TkLightMesonBitWidths::kPtMagSize, AP_RND_CONV, AP_SAT>;
    using glbphi_t     = ap_int<TkLightMesonBitWidths::kGlbPhiSize>;
    using glbeta_t     = ap_int<TkLightMesonBitWidths::kGlbEtaSize>;
    using z0_t         = ap_int<TkLightMesonBitWidths::kZ0Size>;     // 40cm / 0.1
    using mass_t       = ap_ufixed<TkLightMesonBitWidths::kMassSize, TkLightMesonBitWidths::kMassMagSize, AP_RND_CONV, AP_SAT>;
    using type_t       = ap_uint<TkLightMesonBitWidths::kTypeSize>;       //type of meson
    using ntracks_t    = ap_uint<TkLightMesonBitWidths::kNtracksSize>;    //number of tracks
    using index_t      = ap_uint<TkLightMesonBitWidths::kIndexSize>;      //index of track in collection
    using unassigned_t = ap_uint<TkLightMesonBitWidths::kUnassignedSize>; // Unassigned bits

    using tklightmesonword_bs_t = std::bitset<TkLightMesonBitWidths::kTkLightMesonWordSize>;
    using tklightmesonword_t    = ap_uint<TkLightMesonBitWidths::kTkLightMesonWordSize>;

  public:
    // ----------Constructors --------------------------
    TkLightMesonWord() {}
    TkLightMesonWord(valid_t valid, 
		     pt_t pt, 
		     glbphi_t phi, 
		     glbeta_t eta, 
		     z0_t z0, 
		     mass_t mass, 
		     type_t type, 
		     ntracks_t ntracks, 
		     index_t firstIndex, 
		     index_t secondIndex, 
		     unassigned_t unassigned); 
    ~TkLightMesonWord() {}

    // ----------copy constructor ----------------------
    TkLightMesonWord(const TkLightMesonWord& word) { tkLightMesonWord_ = word.tkLightMesonWord_; }

    // ----------operators -----------------------------
    TkLightMesonWord& operator=(const TkLightMesonWord& word) {
      tkLightMesonWord_ = word.tkLightMesonWord_;
      return *this;
    }

    // ----------member functions (getters) ------------
    // These functions return arbitarary precision words (lists of bits) for each quantity
    valid_t validWord() const { return tkLightMesonWord()(TkLightMesonBitLocations::kValidMSB, TkLightMesonBitLocations::kValidLSB); }
    pt_t ptWord() const {
      pt_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kPtMSB, TkLightMesonBitLocations::kPtLSB);
      return ret;
    }
    glbphi_t glbPhiWord() const {
      glbphi_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kGlbPhiMSB, TkLightMesonBitLocations::kGlbPhiLSB);
      return ret;
    }
    glbeta_t glbEtaWord() const {
      glbeta_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kGlbEtaMSB, TkLightMesonBitLocations::kGlbEtaLSB);
      return ret;
    }
    z0_t z0Word() const {
      z0_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kZ0MSB, TkLightMesonBitLocations::kZ0LSB);
      return ret;
    }
    mass_t massWord() const {
      mass_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kMassMSB, TkLightMesonBitLocations::kMassLSB);
      return ret;
    }
    type_t typeWord() const {
      type_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kTypeMSB, TkLightMesonBitLocations::kTypeLSB);
      return ret;
    }
    ntracks_t ntracksWord() const {
      ntracks_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kNtracksMSB, TkLightMesonBitLocations::kNtracksLSB);
      return ret;
    }
    index_t firstIndexWord() const {
      index_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kFirstIndexMSB, TkLightMesonBitLocations::kFirstIndexLSB);
      return ret;
    }
    index_t secondIndexWord() const {
      index_t ret;
      ret.V = tkLightMesonWord()(TkLightMesonBitLocations::kSecondIndexMSB, TkLightMesonBitLocations::kSecondIndexLSB);
      return ret;
    }
    unassigned_t unassignedWord() const {
      return tkLightMesonWord()(TkLightMesonBitLocations::kUnassignedMSB, TkLightMesonBitLocations::kUnassignedLSB);
    }
    tklightmesonword_t tkLightMesonWord() const { return tklightmesonword_t(tkLightMesonWord_.to_string().c_str(), 2); }

    // These functions return the packed bits in integer format for each quantity
    // Signed quantities have the sign enconded in the left-most bit.
    unsigned int validBits() const { return validWord().to_uint(); }
    unsigned int ptBits() const { return ptWord().to_uint(); }
    unsigned int glbPhiBits() const { return glbPhiWord().to_uint(); }
    unsigned int glbEtaBits() const { return glbEtaWord().to_uint(); }
    unsigned int z0Bits() const { return z0Word().to_uint(); }
    unsigned int massBits() const { return massWord().to_uint(); }
    unsigned int typeBits() const { return typeWord().to_uint(); }
    unsigned int ntracksBits() const { return ntracksWord().to_uint(); }
    unsigned int firstIndexBits() const { return firstIndexWord().to_uint(); }
    unsigned int secondIndexBits() const { return secondIndexWord().to_uint(); }
    unsigned int unassignedBits() const { return unassignedWord().to_uint(); }

    // These functions return the unpacked and converted values
    // These functions return real numbers converted from the digitized quantities by unpacking the 64-bit vertex word
    bool valid() const { return validWord().to_bool(); }
    double pt() const { return ptWord().to_double(); }
    double glbphi() const { return glbPhiWord().to_double() * ETAPHI_LSB; }
    double glbeta() const { return glbEtaWord().to_double() * ETAPHI_LSB; }
    double z0() const { return z0Word().to_double() * Z0_LSB; }
    double mass() const { return massWord().to_double(); }
    unsigned int type() const { return typeWord().to_uint(); }
    unsigned int ntracks() const { return ntracksWord().to_uint(); }
    unsigned int firstIndex() const { return firstIndexWord().to_uint(); }
    unsigned int secondIndex() const { return secondIndexWord().to_uint(); }
    unsigned int unassigned() const { return unassignedWord().to_uint(); }

    // ----------member functions (setters) ------------
    void setTkLightMesonWord(valid_t valid, 
			     pt_t pt, 
			     glbphi_t phi,  
			     glbeta_t eta, 
			     z0_t z0, 
			     mass_t mass, 
			     type_t type, 
			     ntracks_t ntracks, 
			     index_t firstIndex, 
			     index_t secondIndex, 
			     unassigned_t unassigned);

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
    tklightmesonword_bs_t tkLightMesonWord_;
  };

  using TkLightMesonWordCollection = std::vector<l1t::TkLightMesonWord>;

}  // namespace l1t
#endif
