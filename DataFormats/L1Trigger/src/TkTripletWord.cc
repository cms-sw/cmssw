// Class to store the 128-bit TkTriplet word for L1 Track Trigger.
// Author: George Krathanasis, CU Boulder (December 2023)

#include "DataFormats/L1Trigger/interface/TkTripletWord.h"

namespace l1t {
  TkTripletWord::TkTripletWord(tktriplet_valid_t valid,
                               tktriplet_pt_t pt,
                               tktriplet_phi_t phi,
                               tktriplet_eta_t eta,
                               tktriplet_mass_t mass,
                               tktriplet_trk_pt_t trk1Pt,
                               tktriplet_trk_pt_t trk2Pt,
                               tktriplet_trk_pt_t trk3Pt,
                               tktriplet_charge_t charge,
                               tktriplet_unassigned_t unassigned) {
    setTkTripletWord(valid, pt, phi, eta, mass, trk1Pt, trk2Pt, trk3Pt, charge, unassigned);
  }

  template <class packVarType>
  inline void TkTripletWord::packIntoWord(unsigned int& currentOffset,
                                          unsigned int wordChunkSize,
                                          packVarType& packVar) {
    for (unsigned int b = currentOffset; b < (currentOffset + wordChunkSize); ++b) {
      tkTripletWord_.set(b, packVar[b - currentOffset]);
    }
    currentOffset += wordChunkSize;
  }

  void TkTripletWord::setTkTripletWord(tktriplet_valid_t valid,
                                       tktriplet_pt_t pt,
                                       tktriplet_phi_t phi,
                                       tktriplet_eta_t eta,
                                       tktriplet_mass_t mass,
                                       tktriplet_trk_pt_t trk1Pt,
                                       tktriplet_trk_pt_t trk2Pt,
                                       tktriplet_trk_pt_t trk3Pt,
                                       tktriplet_charge_t charge,
                                       tktriplet_unassigned_t unassigned) {
    // pack the TkTriplet word
    unsigned int offset = 0;
    packIntoWord(offset, TkTripletBitWidths::kValidSize, valid);
    packIntoWord(offset, TkTripletBitWidths::kPtSize, pt);
    packIntoWord(offset, TkTripletBitWidths::kPhiSize, phi);
    packIntoWord(offset, TkTripletBitWidths::kEtaSize, eta);
    packIntoWord(offset, TkTripletBitWidths::kMassSize, mass);
    packIntoWord(offset, TkTripletBitWidths::kTrk1PtSize, trk1Pt);
    packIntoWord(offset, TkTripletBitWidths::kTrk2PtSize, trk2Pt);
    packIntoWord(offset, TkTripletBitWidths::kTrk3PtSize, trk3Pt);
    packIntoWord(offset, TkTripletBitWidths::kChargeSize, charge);
    packIntoWord(offset, TkTripletBitWidths::kUnassignedSize, unassigned);
  }

}  //namespace l1t
