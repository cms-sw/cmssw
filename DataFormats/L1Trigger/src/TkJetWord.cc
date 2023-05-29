// Class to store the 96-bit TkJet word for L1 Track Trigger.
// Author: Benjamin Radburn-Smith (September 2022)

#include "DataFormats/L1Trigger/interface/TkJetWord.h"

namespace l1t {
  TkJetWord::TkJetWord(pt_t pt,
                       glbeta_t eta,
                       glbphi_t phi,
                       z0_t z0,
                       nt_t nt,
                       nx_t nx,
                       dispflag_t dispflag,
                       tkjetunassigned_t unassigned) {
    setTkJetWord(pt, eta, phi, z0, nt, nx, dispflag, unassigned);
  }

  void TkJetWord::setTkJetWord(pt_t pt,
                               glbeta_t eta,
                               glbphi_t phi,
                               z0_t z0,
                               nt_t nt,
                               nx_t nx,
                               dispflag_t dispflag,
                               tkjetunassigned_t unassigned) {
    // pack the TkJet word
    unsigned int offset = 0;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kPtSize); b++) {
      tkJetWord_.set(b, pt[b - offset]);
    }
    offset += TkJetBitWidths::kPtSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kGlbPhiSize); b++) {
      tkJetWord_.set(b, phi[b - offset]);
    }
    offset += TkJetBitWidths::kGlbPhiSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kGlbEtaSize); b++) {
      tkJetWord_.set(b, eta[b - offset]);
    }
    offset += TkJetBitWidths::kGlbEtaSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kZ0Size); b++) {
      tkJetWord_.set(b, z0[b - offset]);
    }
    offset += TkJetBitWidths::kZ0Size;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kNtSize); b++) {
      tkJetWord_.set(b, nt[b - offset]);
    }
    offset += TkJetBitWidths::kNtSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kXtSize); b++) {
      tkJetWord_.set(b, nx[b - offset]);
    }
    offset += TkJetBitWidths::kXtSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kDispFlagSize); b++) {
      tkJetWord_.set(b, nx[b - offset]);
    }
    offset += TkJetBitWidths::kDispFlagSize;
    for (unsigned int b = offset; b < (offset + TkJetBitWidths::kUnassignedSize); b++) {
      tkJetWord_.set(b, unassigned[b - offset]);
    }
  }

}  //namespace l1t
