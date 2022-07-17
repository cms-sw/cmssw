////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author:      Alexx Perloff
// created:     March 17, 2021
//
///////

#include "DataFormats/L1Trigger/interface/VertexWord.h"

namespace l1t {

  VertexWord::VertexWord(unsigned int valid,
                         double z0,
                         unsigned int multiplicity,
                         double pt,
                         unsigned int quality,
                         unsigned int inverseMultiplicity,
                         unsigned int unassigned) {
    // convert directly to AP types
    vtxvalid_t valid_ap = valid;
    vtxz0_t z0_ap = z0;
    vtxmultiplicity_t mult_ap = multiplicity;
    vtxsumpt_t pt_ap = pt;
    vtxquality_t quality_ap = quality;
    vtxinversemult_t invmult_ap = inverseMultiplicity;
    vtxunassigned_t unassigned_ap = unassigned;

    setVertexWord(valid_ap, z0_ap, mult_ap, pt_ap, quality_ap, invmult_ap, unassigned_ap);
  }

  VertexWord::VertexWord(unsigned int valid,
                         unsigned int z0,
                         unsigned int multiplicity,
                         unsigned int pt,
                         unsigned int quality,
                         unsigned int inverseMultiplicity,
                         unsigned int unassigned) {
    // convert to AP types
    vtxvalid_t valid_ap = valid;
    vtxz0_t z0_ap = unpackSignedValue(
        z0, VertexBitWidths::kZ0Size, 1.0 / (1 << (VertexBitWidths::kZ0Size - VertexBitWidths::kZ0MagSize)));
    vtxmultiplicity_t mult_ap = multiplicity;
    vtxsumpt_t pt_ap = unpackSignedValue(
        z0, VertexBitWidths::kSumPtSize, 1.0 / (1 << (VertexBitWidths::kSumPtSize - VertexBitWidths::kSumPtMagSize)));
    vtxquality_t quality_ap = quality;
    vtxinversemult_t invmult_ap = inverseMultiplicity;
    vtxunassigned_t unassigned_ap = unassigned;

    setVertexWord(valid_ap, z0_ap, mult_ap, pt_ap, quality_ap, invmult_ap, unassigned_ap);
  }

  VertexWord::VertexWord(vtxvalid_t valid,
                         vtxz0_t z0,
                         vtxmultiplicity_t multiplicity,
                         vtxsumpt_t pt,
                         vtxquality_t quality,
                         vtxinversemult_t inverseMultiplicity,
                         vtxunassigned_t unassigned) {
    setVertexWord(valid, z0, multiplicity, pt, quality, inverseMultiplicity, unassigned);
  }

  void VertexWord::setVertexWord(vtxvalid_t valid,
                                 vtxz0_t z0,
                                 vtxmultiplicity_t multiplicity,
                                 vtxsumpt_t pt,
                                 vtxquality_t quality,
                                 vtxinversemult_t inverseMultiplicity,
                                 vtxunassigned_t unassigned) {
    // pack the vertex word
    unsigned int offset = 0;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kValidSize); b++) {
      vertexWord_.set(b, valid[b - offset]);
    }
    offset += VertexBitWidths::kValidSize;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kZ0Size); b++) {
      vertexWord_.set(b, z0[b - offset]);
    }
    offset += VertexBitWidths::kZ0Size;

    for (unsigned int b = offset; b < (offset + VertexBitWidths::kNTrackInPVSize); b++) {
      vertexWord_.set(b, multiplicity[b - offset]);
    }
    offset += VertexBitWidths::kNTrackInPVSize;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kSumPtSize); b++) {
      vertexWord_.set(b, pt[b - offset]);
    }
    offset += VertexBitWidths::kSumPtSize;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kQualitySize); b++) {
      vertexWord_.set(b, quality[b - offset]);
    }
    offset += VertexBitWidths::kQualitySize;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kNTrackOutPVSize); b++) {
      vertexWord_.set(b, inverseMultiplicity[b - offset]);
    }
    offset += VertexBitWidths::kNTrackOutPVSize;
    for (unsigned int b = offset; b < (offset + VertexBitWidths::kUnassignedSize); b++) {
      vertexWord_.set(b, unassigned[b - offset]);
    }
  }

}  // namespace l1t