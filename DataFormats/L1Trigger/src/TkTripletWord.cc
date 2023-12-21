// Class to store the 128-bit TkTriplet word for L1 Track Trigger.
// Author: George Krathanasis, CU Boulder (December 2023)

#include "DataFormats/L1Trigger/interface/TkTripletWord.h"

namespace l1t {
  TkTripletWord::TkTripletWord(valid_t valid,
                               pt_t pt,
                               glbeta_t eta,
                               glbphi_t phi,
                               mass_t mass,
                               charge_t charge,
                               ditrack_minmass_t ditrack_minmass,
                               ditrack_maxmass_t ditrack_maxmass,
                               ditrack_minz0_t ditrack_minz0,
                               ditrack_maxz0_t ditrack_maxz0,
                               unassigned_t unassigned) {
    setTkTripletWord(
        valid, pt, eta, phi, mass, charge, ditrack_minmass, ditrack_maxmass, ditrack_minz0, ditrack_maxz0, unassigned);
  }

  void TkTripletWord::setTkTripletWord(valid_t valid,
                                       pt_t pt,
                                       glbeta_t eta,
                                       glbphi_t phi,
                                       mass_t mass,
                                       charge_t charge,
                                       ditrack_minmass_t ditrack_minmass,
                                       ditrack_maxmass_t ditrack_maxmass,
                                       ditrack_minz0_t ditrack_minz0,
                                       ditrack_maxz0_t ditrack_maxz0,
                                       unassigned_t unassigned) {
    // pack the TkTriplet word
    unsigned int offset = 0;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kValidSize); b++) {
      tkTripletWord_.set(b, valid[b - offset]);
    }
    offset += TkTripletBitWidths::kValidSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kPtSize); b++) {
      tkTripletWord_.set(b, pt[b - offset]);
    }
    offset += TkTripletBitWidths::kPtSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kGlbPhiSize); b++) {
      tkTripletWord_.set(b, phi[b - offset]);
    }
    offset += TkTripletBitWidths::kGlbPhiSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kGlbEtaSize); b++) {
      tkTripletWord_.set(b, eta[b - offset]);
    }
    offset += TkTripletBitWidths::kGlbEtaSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kMassSize); b++) {
      tkTripletWord_.set(b, mass[b - offset]);
    }
    offset += TkTripletBitWidths::kMassSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kChargeSize); b++) {
      tkTripletWord_.set(b, charge[b - offset]);
    }
    offset += TkTripletBitWidths::kChargeSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kDiTrackMinMassSize); b++) {
      tkTripletWord_.set(b, ditrack_minmass[b - offset]);
    }
    offset += TkTripletBitWidths::kDiTrackMinMassSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kDiTrackMaxMassSize); b++) {
      tkTripletWord_.set(b, ditrack_maxmass[b - offset]);
    }
    offset += TkTripletBitWidths::kDiTrackMaxMassSize;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kDiTrackMinZ0Size); b++) {
      tkTripletWord_.set(b, ditrack_minz0[b - offset]);
    }
    offset += TkTripletBitWidths::kDiTrackMinZ0Size;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kDiTrackMaxZ0Size); b++) {
      tkTripletWord_.set(b, ditrack_maxz0[b - offset]);
    }
    offset += TkTripletBitWidths::kDiTrackMaxZ0Size;
    for (unsigned int b = offset; b < (offset + TkTripletBitWidths::kUnassignedSize); b++) {
      tkTripletWord_.set(b, unassigned[b - offset]);
    }
  }

}  //namespace l1t
