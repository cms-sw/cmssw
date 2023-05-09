#ifndef DataFormats_L1TVertex_VertexWord_h
#define DataFormats_L1TVertex_VertexWord_h

////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author:      Alexx Perloff
// created:     March 17, 2021
// modified by:    Nick Manganelli
// modified:    November 18, 2022
//
///////

#include "DataFormats/L1Trigger/interface/Vertex.h"

#include <ap_int.h>
#include <ap_fixed.h>

#include <bitset>
#include <vector>

namespace l1t {

  class VertexWord {
  public:
    // ----------constants, enums and typedefs ---------
    enum VertexBitWidths {
      // The sizes of the vertex word components and total word size
      kValidSize = 1,         // Width of the valid bit
      kZ0Size = 15,           // Width of z-position
      kZ0MagSize = 6,         // Width of z-position magnitude (signed)
      kNTrackInPVSize = 8,    // Width of the multiplicity in the PV (unsigned)
      kSumPtSize = 12,        // Width of pT
      kSumPtMagSize = 10,     // Width of pT magnitude (unsigned)
      kQualitySize = 3,       // Width of the quality field
      kNTrackOutPVSize = 10,  // Width of the multiplicity outside the PV (unsigned)
      kUnassignedSize = 15,   // Width of the unassigned bits

      kVertexWordSize = kValidSize + kZ0Size + kNTrackInPVSize + kSumPtSize + kQualitySize + kNTrackOutPVSize +
                        kUnassignedSize,  // Width of the vertex word in bits
    };

    enum VertexBitLocations {
      // The location of the least significant bit (LSB) and most significant bit (MSB) in the vertex word for different fields
      kValidLSB = 0,
      kValidMSB = kValidLSB + VertexBitWidths::kValidSize - 1,
      kZ0LSB = kValidMSB + 1,
      kZ0MSB = kZ0LSB + VertexBitWidths::kZ0Size - 1,
      kNTrackInPVLSB = kZ0MSB + 1,
      kNTrackInPVMSB = kNTrackInPVLSB + VertexBitWidths::kNTrackInPVSize - 1,
      kSumPtLSB = kNTrackInPVMSB + 1,
      kSumPtMSB = kSumPtLSB + VertexBitWidths::kSumPtSize - 1,
      kQualityLSB = kSumPtMSB + 1,
      kQualityMSB = kQualityLSB + VertexBitWidths::kQualitySize - 1,
      kNTrackOutPVLSB = kQualityMSB + 1,
      kNTrackOutPVMSB = kNTrackOutPVLSB + VertexBitWidths::kNTrackOutPVSize - 1,
      kUnassignedLSB = kNTrackOutPVMSB + 1,
      kUnassignedMSB = kUnassignedLSB + VertexBitWidths::kUnassignedSize - 1
    };

    // vertex parameters types
    typedef ap_uint<VertexBitWidths::kValidSize> vtxvalid_t;  // Vertex validity
    typedef ap_fixed<VertexBitWidths::kZ0Size, VertexBitWidths::kZ0MagSize, AP_RND_CONV, AP_SAT> vtxz0_t;  // Vertex z0
    typedef ap_ufixed<VertexBitWidths::kNTrackInPVSize, VertexBitWidths::kNTrackInPVSize, AP_RND_CONV, AP_SAT>
        vtxmultiplicity_t;  // NTracks in vertex
    typedef ap_ufixed<VertexBitWidths::kSumPtSize, VertexBitWidths::kSumPtMagSize, AP_RND_CONV, AP_SAT>
        vtxsumpt_t;                                               // Vertex Sum pT
    typedef ap_uint<VertexBitWidths::kQualitySize> vtxquality_t;  // Vertex quality
    typedef ap_ufixed<VertexBitWidths::kNTrackOutPVSize, VertexBitWidths::kNTrackOutPVSize, AP_RND_CONV, AP_SAT>
        vtxinversemult_t;                                               // NTracks outside vertex
    typedef ap_uint<VertexBitWidths::kUnassignedSize> vtxunassigned_t;  // Unassigned bits

    // vertex word types
    typedef std::bitset<VertexBitWidths::kVertexWordSize> vtxword_bs_t;  // Entire track word;
    typedef ap_uint<VertexBitWidths::kVertexWordSize> vtxword_t;         // Entire vertex word;

    // reference types
    typedef edm::Ref<l1t::VertexCollection> VertexRef;  // Reference to a persistent l1t::Vertex

  public:
    // ----------Constructors --------------------------
    VertexWord() {}
    VertexWord(unsigned int valid,
               double z0,
               unsigned int multiplicity,
               double pt,
               unsigned int quality,
               unsigned int inverseMultiplicity,
               unsigned int unassigned);
    VertexWord(unsigned int valid,
               unsigned int z0,
               unsigned int multiplicity,
               unsigned int pt,
               unsigned int quality,
               unsigned int inverseMultiplicity,
               unsigned int unassigned);
    VertexWord(vtxvalid_t valid,
               vtxz0_t z0,
               vtxmultiplicity_t multiplicity,
               vtxsumpt_t pt,
               vtxquality_t quality,
               vtxinversemult_t inverseMultiplicity,
               vtxunassigned_t unassigned);

    ~VertexWord() {}

    // ----------copy constructor ----------------------
    VertexWord(const VertexWord& word) { vertexWord_ = word.vertexWord_; }

    // ----------operators -----------------------------
    VertexWord& operator=(const VertexWord& word) {
      vertexWord_ = word.vertexWord_;
      return *this;
    }

    // ----------member functions (getters) ------------
    // These functions return arbitarary precision words (lists of bits) for each quantity
    vtxvalid_t validWord() const { return vertexWord()(VertexBitLocations::kValidMSB, VertexBitLocations::kValidLSB); }
    vtxz0_t z0Word() const {
      vtxz0_t ret;
      ret.V = vertexWord()(VertexBitLocations::kZ0MSB, VertexBitLocations::kZ0LSB);
      return ret;
    }
    vtxmultiplicity_t multiplicityWord() const {
      return vertexWord()(VertexBitLocations::kNTrackInPVMSB, VertexBitLocations::kNTrackInPVLSB);
    }
    vtxsumpt_t ptWord() const {
      vtxsumpt_t ret;
      ret.V = vertexWord()(VertexBitLocations::kSumPtMSB, VertexBitLocations::kSumPtLSB);
      return ret;
    }
    vtxquality_t qualityWord() const {
      return vertexWord()(VertexBitLocations::kQualityMSB, VertexBitLocations::kQualityLSB);
    }
    vtxinversemult_t inverseMultiplicityWord() const {
      return vertexWord()(VertexBitLocations::kNTrackOutPVMSB, VertexBitLocations::kNTrackOutPVLSB);
    }
    vtxunassigned_t unassignedWord() const {
      return vertexWord()(VertexBitLocations::kUnassignedMSB, VertexBitLocations::kUnassignedLSB);
    }
    vtxword_t vertexWord() const { return vtxword_t(vertexWord_.to_string().c_str(), 2); }

    // These functions return the packed bits in integer format for each quantity
    // Signed quantities have the sign enconded in the left-most bit.
    unsigned int validBits() const { return validWord().to_uint(); }
    unsigned int z0Bits() const { return z0Word().to_uint(); }
    unsigned int multiplicityBits() const { return multiplicityWord().to_uint(); }
    unsigned int ptBits() const { return ptWord().to_uint(); }
    unsigned int qualityBits() const { return qualityWord().to_uint(); }
    unsigned int inverseMultiplicityBits() const { return inverseMultiplicityWord().to_uint(); }
    unsigned int unassignedBits() const { return unassignedWord().to_uint(); }

    // These functions return the unpacked and converted values
    // These functions return real numbers converted from the digitized quantities by unpacking the 64-bit vertex word
    bool valid() const { return validWord().to_bool(); }
    double z0() const { return z0Word().to_double(); }
    unsigned int multiplicity() const { return multiplicityWord().to_uint(); }
    double pt() const { return ptWord().to_double(); }
    unsigned int quality() const { return qualityWord().to_uint(); }
    unsigned int inverseMultiplicity() const { return inverseMultiplicityWord().to_uint(); }
    unsigned int unassigned() const { return unassignedWord().to_uint(); }

    // return reference to floating point vertex
    VertexRef vertexRef() const { return vertexRef_; }

    // ----------member functions (setters) ------------
    void setVertexWord(vtxvalid_t valid,
                       vtxz0_t z0,
                       vtxmultiplicity_t multiplicity,
                       vtxsumpt_t pt,
                       vtxquality_t quality,
                       vtxinversemult_t inverseMultiplicity,
                       vtxunassigned_t unassigned);

    // set reference to floating point vertex
    void setVertexRef(const VertexRef& ref) { vertexRef_ = ref; }

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
    vtxword_bs_t vertexWord_;
    VertexRef vertexRef_;
  };

  typedef std::vector<VertexWord> VertexWordCollection;
  typedef edm::Ref<VertexWordCollection> VertexWordRef;
}  // namespace l1t

#endif
