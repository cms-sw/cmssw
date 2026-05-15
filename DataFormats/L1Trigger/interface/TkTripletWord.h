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
  namespace io_v1 {
    class TkTripletWord {
    public:
      // ----------constants, enums and typedefs ---------
      static constexpr double MAX_MASS = 1000.;
      static constexpr double MAX_ETA = 8.;
      static constexpr double MAX_CHARGE = 3.;
      static constexpr double MAX_Z0 = 25.;

      enum TkTripletBitWidths {
        // The sizes of the triplet word components and total word size
        kValidSize = 1,  // Width of the valid bit
        kPtSize = 16,    // Width of the triplet pt
        kPtMagSize = 11,
        kGlbEtaSize = 14,          // Width of the triplet eta
        kGlbPhiSize = 13,          // Width of the triplet phi
        kMassSize = 16,            // Width of the triplet mass
        kChargeSize = 3,           // Width of the triplet charge
        kDiTrackMinMassSize = 16,  // Width of the mass of min mass pair
        kDiTrackMaxMassSize = 16,  // Width of the mass of max mass pair
        kDiTrackMinZ0Size = 8,     // Width of the Dz of min Dz pair
        kDiTrackMaxZ0Size = 8,     // Width of the Dz of max Dz pair
        kUnassignedSize = 17,
        kTkTripletWordSize = kValidSize + kPtSize + kGlbEtaSize + kGlbPhiSize + kMassSize + kChargeSize +
                             kDiTrackMinMassSize + kDiTrackMaxMassSize + kDiTrackMinZ0Size + kDiTrackMaxZ0Size +
                             kUnassignedSize,
      };

      enum TkTripletBitLocations {
        // The location of the least significant bit (LSB) and most significant bit (MSB) in the triplet word for different fields
        kValidLSB = 0,
        kValidMSB = kValidLSB + TkTripletBitWidths::kValidSize - 1,
        kPtLSB = kValidMSB + 1,
        kPtMSB = kPtLSB + TkTripletBitWidths::kPtSize - 1,
        kGlbPhiLSB = kPtMSB + 1,
        kGlbPhiMSB = kGlbPhiLSB + TkTripletBitWidths::kGlbPhiSize - 1,
        kGlbEtaLSB = kGlbPhiMSB + 1,
        kGlbEtaMSB = kGlbEtaLSB + TkTripletBitWidths::kGlbEtaSize - 1,
        kMassLSB = kGlbEtaMSB + 1,
        kMassMSB = kMassLSB + TkTripletBitWidths::kMassSize - 1,
        kChargeLSB = kMassMSB + 1,
        kChargeMSB = kChargeLSB + TkTripletBitWidths::kChargeSize - 1,
        kDiTrackMinMassLSB = kChargeMSB + 1,
        kDiTrackMinMassMSB = kDiTrackMinMassLSB + TkTripletBitWidths::kDiTrackMinMassSize - 1,
        kDiTrackMaxMassLSB = kDiTrackMinMassMSB + 1,
        kDiTrackMaxMassMSB = kDiTrackMaxMassLSB + TkTripletBitWidths::kDiTrackMaxMassSize - 1,
        kDiTrackMinZ0LSB = kDiTrackMaxMassMSB + 1,
        kDiTrackMinZ0MSB = kDiTrackMinZ0LSB + TkTripletBitWidths::kDiTrackMinZ0Size - 1,
        kDiTrackMaxZ0LSB = kDiTrackMinZ0MSB + 1,
        kDiTrackMaxZ0MSB = kDiTrackMaxZ0LSB + TkTripletBitWidths::kDiTrackMaxZ0Size - 1,
        kUnassignedLSB = kDiTrackMaxZ0MSB + 1,
        kUnassignedMSB = kUnassignedLSB + TkTripletBitWidths::kUnassignedSize - 1,
      };

      // vertex parameters types
      typedef ap_uint<kValidSize> valid_t;                          //valid
      typedef ap_ufixed<kPtSize, kPtMagSize, AP_TRN, AP_SAT> pt_t;  //triplet pt
      typedef ap_int<kGlbEtaSize> glbeta_t;                         //triplet eta
      typedef ap_int<kGlbPhiSize> glbphi_t;                         //triplet phi
      typedef ap_int<kMassSize> mass_t;                             //triplet mass
      typedef ap_int<kChargeSize> charge_t;                         //triplet Q
      typedef ap_int<kDiTrackMinMassSize> ditrack_minmass_t;        //pair min mass
      typedef ap_int<kDiTrackMaxMassSize> ditrack_maxmass_t;        //pair max mass
      typedef ap_int<kDiTrackMinZ0Size> ditrack_minz0_t;            //pair dz min
      typedef ap_int<kDiTrackMaxZ0Size> ditrack_maxz0_t;            //pair dz max
      typedef ap_uint<TkTripletBitWidths::kUnassignedSize> unassigned_t;
      typedef std::bitset<TkTripletBitWidths::kTkTripletWordSize> tktripletword_bs_t;
      typedef ap_uint<TkTripletBitWidths::kTkTripletWordSize> tktripletword_t;

    public:
      // ----------Constructors --------------------------
      TkTripletWord() {}
      TkTripletWord(valid_t valid,
                    pt_t pt,
                    glbeta_t eta,
                    glbphi_t phi,
                    mass_t mass,
                    charge_t charge,
                    ditrack_minmass_t ditrack_minmass,
                    ditrack_maxmass_t ditrack_maxmass,
                    ditrack_minz0_t ditrack_minz0_t,
                    ditrack_maxz0_t ditrack_maxz0_t,
                    unassigned_t unassigned);

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
      valid_t validWord() const {
        return tkTripletWord()(TkTripletBitLocations::kValidMSB, TkTripletBitLocations::kValidLSB);
      }
      pt_t ptWord() const {
        pt_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kPtMSB, TkTripletBitLocations::kPtLSB);
        return ret;
      }
      glbeta_t glbEtaWord() const {
        glbeta_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kGlbEtaMSB, TkTripletBitLocations::kGlbEtaLSB);
        return ret;
      }
      glbphi_t glbPhiWord() const {
        glbphi_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kGlbPhiMSB, TkTripletBitLocations::kGlbPhiLSB);
        return ret;
      }
      mass_t massWord() const {
        mass_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kMassMSB, TkTripletBitLocations::kMassLSB);
        return ret;
      }
      charge_t chargeWord() const {
        charge_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kChargeMSB, TkTripletBitLocations::kChargeLSB);
        return ret;
      }

      ditrack_minmass_t ditrackMinMassWord() const {
        ditrack_minmass_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kDiTrackMinMassMSB, TkTripletBitLocations::kDiTrackMinMassLSB);
        return ret;
      }
      ditrack_maxmass_t ditrackMaxMassWord() const {
        ditrack_maxmass_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kDiTrackMaxMassMSB, TkTripletBitLocations::kDiTrackMaxMassLSB);
        return ret;
      }
      ditrack_minz0_t ditrackMinZ0Word() const {
        ditrack_minz0_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kDiTrackMinZ0MSB, TkTripletBitLocations::kDiTrackMinZ0LSB);
        return ret;
      }
      ditrack_maxz0_t ditrackMaxZ0Word() const {
        ditrack_maxz0_t ret;
        ret.V = tkTripletWord()(TkTripletBitLocations::kDiTrackMaxZ0MSB, TkTripletBitLocations::kDiTrackMaxZ0LSB);
        return ret;
      }
      unassigned_t unassignedWord() const {
        return tkTripletWord()(TkTripletBitLocations::kUnassignedMSB, TkTripletBitLocations::kUnassignedLSB);
      }
      tktripletword_t tkTripletWord() const { return tktripletword_t(tkTripletWord_.to_string().c_str(), 2); }

      // These functions return the packed bits in integer format for each quantity
      // Signed quantities have the sign enconded in the left-most bit.
      unsigned int validBits() const { return validWord().to_uint(); }
      unsigned int ptBits() const { return ptWord().to_uint(); }
      unsigned int glbEtaBits() const { return glbEtaWord().to_uint(); }
      unsigned int glbPhiBits() const { return glbPhiWord().to_uint(); }
      unsigned int massBits() const { return massWord().to_uint(); }
      unsigned int chargeBits() const { return chargeWord().to_uint(); }
      unsigned int ditrackMinMassBits() const { return ditrackMinMassWord().to_uint(); }
      unsigned int ditrackMaxMassBits() const { return ditrackMaxMassWord().to_uint(); }
      unsigned int ditrackMinZ0Bits() const { return ditrackMinZ0Word().to_uint(); }
      unsigned int ditrackMaxZ0Bits() const { return ditrackMaxZ0Word().to_uint(); }
      unsigned int unassignedBits() const { return unassignedWord().to_uint(); }

      // These functions return the unpacked and converted values
      // These functions return real numbers converted from the digitized quantities by unpacking the 64-bit vertex word
      bool valid() const { return validWord().to_bool(); }
      float pt() const { return ptWord().to_float(); }
      float glbeta() const {
        return unpackSignedValue(
            glbEtaWord(), TkTripletBitWidths::kGlbEtaSize, (MAX_ETA) / (1 << TkTripletBitWidths::kGlbEtaSize));
      }
      float glbphi() const {
        return unpackSignedValue(glbPhiWord(),
                                 TkTripletBitWidths::kGlbPhiSize,
                                 (2. * std::abs(M_PI)) / (1 << TkTripletBitWidths::kGlbPhiSize));
      }
      float mass() const {
        return unpackSignedValue(
            massWord(), TkTripletBitWidths::kMassSize, MAX_MASS / (1 << TkTripletBitWidths::kMassSize));
      }
      int charge() const {
        return unpackSignedValue(
            chargeWord(), TkTripletBitWidths::kChargeSize, MAX_CHARGE / (1 << TkTripletBitWidths::kChargeSize));
      }
      float ditrackMinMass() const {
        return unpackSignedValue(ditrackMinMassWord(),
                                 TkTripletBitWidths::kDiTrackMinMassSize,
                                 MAX_MASS / (1 << TkTripletBitWidths::kDiTrackMinMassSize));
      }
      float ditrackMaxMass() const {
        return unpackSignedValue(ditrackMaxMassWord(),
                                 TkTripletBitWidths::kDiTrackMaxMassSize,
                                 MAX_MASS / (1 << TkTripletBitWidths::kDiTrackMaxMassSize));
      }
      float ditrackMinZ0() const {
        return unpackSignedValue(ditrackMinZ0Word(),
                                 TkTripletBitWidths::kDiTrackMinZ0Size,
                                 MAX_Z0 / (1 << TkTripletBitWidths::kDiTrackMinZ0Size));
      }
      float ditrackMaxZ0() const {
        return unpackSignedValue(ditrackMaxZ0Word(),
                                 TkTripletBitWidths::kDiTrackMaxZ0Size,
                                 MAX_Z0 / (1 << TkTripletBitWidths::kDiTrackMaxZ0Size));
      }
      unsigned int unassigned() const { return unassignedWord().to_uint(); }

      // ----------member functions (setters) ------------
      void setTkTripletWord(valid_t valid,
                            pt_t pt,
                            glbeta_t eta,
                            glbphi_t phi,
                            mass_t mass,
                            charge_t charge,
                            ditrack_minmass_t ditrack_minmass,
                            ditrack_maxmass_t ditrack_maxmass,
                            ditrack_minz0_t ditrack_minz0,
                            ditrack_maxz0_t ditrack_maxz0,
                            unassigned_t unassigned);

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
  }  // namespace io_v1
  using TkTripletWord = io_v1::TkTripletWord;
  typedef std::vector<l1t::TkTripletWord> TkTripletWordCollection;

}  // namespace l1t

#endif
