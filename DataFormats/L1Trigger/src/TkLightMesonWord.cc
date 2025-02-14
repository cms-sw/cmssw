///////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//----------------------------------------------------------------------------
// Authors: Alexx Perloff, Pritam Palit (original version, 2021),
//          Sweta Baradia, Suchandra Dutta, Subir Sarkar (February 2025)
//----------------------------------------------------------------------------

#include "DataFormats/L1Trigger/interface/TkLightMesonWord.h"

namespace l1t {

  TkLightMesonWord::TkLightMesonWord(
      valid_t valid,
      pt_t pt,
      glbphi_t phi,
      glbeta_t eta,
      z0_t z0,
      mass_t mass,
      type_t type,
      ntracks_t ntracks,
      index_t
          firstIndex,  //Index to store the the position of first selected track from Phi (or first selected Phi) in pos track collection (Phi collection)
      index_t
          secondIndex,  //Index to store the the position of second selected track from Phi (or first selected Phi) in neg track collection (Phi collection)
      unassigned_t unassigned) {
    setTkLightMesonWord(valid, pt, phi, eta, z0, mass, type, ntracks, firstIndex, secondIndex, unassigned);
  }
  void TkLightMesonWord::setTkLightMesonWord(valid_t valid,
                                             pt_t pt,
                                             glbphi_t phi,
                                             glbeta_t eta,
                                             z0_t z0,
                                             mass_t mass,
                                             type_t type,
                                             ntracks_t ntracks,
                                             index_t firstIndex,
                                             index_t secondIndex,
                                             unassigned_t unassigned) {
    // pack the TkLightMesonWord
    unsigned int offset = 0;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kValidSize; b++) {
      tkLightMesonWord_.set(b, valid[b - offset]);
    }
    offset += TkLightMesonBitWidths::kValidSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kPtSize; b++) {
      tkLightMesonWord_.set(b, pt[b - offset]);
    }
    offset += TkLightMesonBitWidths::kPtSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kGlbPhiSize; b++) {
      tkLightMesonWord_.set(b, phi[b - offset]);
    }
    offset += TkLightMesonBitWidths::kGlbPhiSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kGlbEtaSize; b++) {
      tkLightMesonWord_.set(b, eta[b - offset]);
    }
    offset += TkLightMesonBitWidths::kGlbEtaSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kZ0Size; b++) {
      tkLightMesonWord_.set(b, z0[b - offset]);
    }
    offset += TkLightMesonBitWidths::kZ0Size;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kMassSize; b++) {
      tkLightMesonWord_.set(b, mass[b - offset]);
    }
    offset += TkLightMesonBitWidths::kMassSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kTypeSize; b++) {
      tkLightMesonWord_.set(b, type[b - offset]);
    }
    offset += TkLightMesonBitWidths::kTypeSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kNtracksSize; b++) {
      tkLightMesonWord_.set(b, ntracks[b - offset]);
    }
    offset += TkLightMesonBitWidths::kNtracksSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kIndexSize; b++) {
      tkLightMesonWord_.set(b, firstIndex[b - offset]);
    }
    offset += TkLightMesonBitWidths::kIndexSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kIndexSize; b++) {
      tkLightMesonWord_.set(b, secondIndex[b - offset]);
    }
    offset += TkLightMesonBitWidths::kIndexSize;

    for (unsigned int b = offset; b < offset + TkLightMesonBitWidths::kUnassignedSize; b++) {
      tkLightMesonWord_.set(b, unassigned[b - offset]);
    }
  }
}  // namespace l1t
