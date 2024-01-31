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

  template <class packVarType>
  inline void TkTripletWord::packIntoWord(unsigned int& currentOffset,
                                          unsigned int wordChunkSize,
                                          packVarType& packVar) {
    for (unsigned int b = currentOffset; b < (currentOffset + wordChunkSize); ++b) {
      tkTripletWord_.set(b, packVar[b - currentOffset]);
    }
    currentOffset += wordChunkSize;
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
    packIntoWord(offset, TkTripletBitWidths::kValidSize, valid);
    packIntoWord(offset, TkTripletBitWidths::kPtSize, pt);
    packIntoWord(offset, TkTripletBitWidths::kGlbPhiSize, phi);
    packIntoWord(offset, TkTripletBitWidths::kGlbEtaSize, eta);
    packIntoWord(offset, TkTripletBitWidths::kMassSize, mass);
    packIntoWord(offset, TkTripletBitWidths::kChargeSize, charge);
    packIntoWord(offset, TkTripletBitWidths::kDiTrackMinMassSize, ditrack_minmass);
    packIntoWord(offset, TkTripletBitWidths::kDiTrackMaxMassSize, ditrack_maxmass);
    packIntoWord(offset, TkTripletBitWidths::kDiTrackMinZ0Size, ditrack_minz0);
    packIntoWord(offset, TkTripletBitWidths::kDiTrackMaxZ0Size, ditrack_maxz0);
  }

}  //namespace l1t
