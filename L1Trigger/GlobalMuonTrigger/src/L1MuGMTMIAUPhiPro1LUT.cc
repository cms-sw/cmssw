//-------------------------------------------------
//
//   Class: L1MuGMTMIAUPhiPro1LUT
//
//
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro1LUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPhiLUT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUPhiPro1LUT::InitParameters() {
  m_calo_align = 0.;  //***FIXME: read from DB or .orcarc
}

//--------------------------------------------------------------------------------
// Phi Projection LUT 1: project in phi based on eta(4bit), pt and charge
// =====================
//
// This LUT contains the phi projection. Based on the fine phi (lower 3 bits of the
// phi code) inside a calo region, eta(4bit), pt and charge it delivers a 3-bit offset
// to the starting calo region(=upper 5 bits of phi code) and a 1-bit fine grain
// information indicating which half of the target calo region the projection points to.
//
// The offset for positive charge is always positive, for neg charge it is always negative
// However, to accomodate for a calo alignment, an offset of 1 region into the opposite
// direction is possible.
//
// meaning of 3-bit calo-offset cphi_ofs:
//        charge +               charge -
//       offset        phi(*)     offset      phi(*)
// 0   | -1 reg  -20deg..  0deg | +1 reg  +20deg..  40deg |
// 1   | 0       - 0deg.. 20deg | 0         0deg..  20deg |
// 2   | +1 reg   20deg.. 40deg | -1 reg  -20deg..   0deg |
// 3   | +2 reg   40deg.. 60deg | -2 reg  -40deg.. -20deg |
// 4   | +3 reg   60deg.. 80deg | -3 reg  -60deg.. -40deg |
// 5   | +4 reg   80deg..100deg | -4 reg  -80deg.. -60deg |
// 6   | +5 reg  100deg..120deg | -5 reg -100deg.. -80deg |
// 7   | +6 reg  120deg..140deg | -6 reg -120deg..-100deg |
//
// (*) relative to low edge of start region
//
//
//
// It is possible to specify a calo alignment: This is the angle between the low
// edge of the first calo region an the low edge of the first bin of the muon phi
// scale. By default both low edges are at phi=0.
//
// calo_align = Phi(low edge of first calo reg) - Phi (Low edge of first muon oh bin)
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTMIAUPhiPro1LUT::TheLookupFunction(
    int idx, unsigned phi_fine, unsigned eta, unsigned pt, unsigned charge) const {
  // idx is MIP_DT, MIP_BRPC, ISO_DT, ISO_BRPC, MIP_CSC, MIP_FRPC, ISO_CSC, ISO_FRPC
  // INPUTS:  phi_fine(3) eta(4) pt(5) charge(1)
  // OUTPUTS: cphi_fine(1) cphi_ofs(3)

  //  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();
  const L1MuTriggerPtScale* theTriggerPtScale = L1MuGMTConfig::getTriggerPtScale();

  int isRPC = idx % 2;
  int isFWD = idx / 4;

  int isys = isFWD + 2 * isRPC;  // DT, CSC, BRPC, FRPC
  int isISO = (idx / 2) % 2;

  int ch_idx = (charge == 0) ? 1 : 0;  // positive charge is 0 (but idx 1)

  // currently only support 3-bit eta (3 lower bits); ignore 4th bit
  if (eta > 7)
    eta -= 8;

  float dphi = L1MuGMTPhiLUT::dphi(isys,
                                   isISO,
                                   ch_idx,
                                   (int)eta,
                                   theTriggerPtScale->getPtScale()->getLowEdge(pt));  // use old LUT, here
  // theTriggerScales->getPtScale()->getLowEdge(pt) );  // use old LUT, here

  // calculate phi in calo relative to low edge of start region
  // == use low edge of muon phi bin as dphi was calculated with this assumption

  float calophi = phi_fine * 2.5 / 180. * M_PI - dphi - m_calo_align;

  if (charge == 0 && calophi < 0.) {  // plus charge
    edm::LogWarning("LUTMismatch")
        << "warning: calo offset goes into wrong direction. charge is plus and calophi < 0deg" << endl
        << "SYS="
        << (isys == 0   ? "DT"
            : isys == 1 ? "CSC"
            : isys == 2 ? "BRPC"
                        : "FRPC")
        << " ISO = " << isISO << " etabin = " << eta << " pval = "
        << theTriggerPtScale->getPtScale()->getLowEdge(pt)
        // << " pval = " << theTriggerScales->getPtScale()->getLowEdge(pt)
        << " charge = " << (charge == 0 ? "pos" : "neg") << " phi_fine = " << phi_fine
        << " calophi(deg) = " << calophi * 180. / M_PI << endl;
  } else if (charge == 1 && calophi > 20. / 180. * M_PI) {  // neg charge
    edm::LogWarning("LUTMismatch")
        << "warning: calo offset goes into wrong direction. charge is minus and calophi > 20deg" << endl
        << "SYS="
        << (isys == 0   ? "DT"
            : isys == 1 ? "CSC"
            : isys == 2 ? "BRPC"
                        : "FRPC")
        << " ISO = " << isISO << " etabin = " << eta << " pval = "
        << theTriggerPtScale->getPtScale()->getLowEdge(pt)
        // << " pval = " << theTriggerScales->getPtScale()->getLowEdge(pt)
        << " charge = " << (charge == 0 ? "pos" : "neg") << " phi_fine = " << phi_fine
        << " calophi(deg) = " << calophi * 180. / M_PI << endl;
  }

  // which half of calo region
  int cphi_fine = (int)((calophi + 2. * M_PI) / (10. / 180. * M_PI));
  cphi_fine %= 2;

  // shift by one region so that an offset in wrong direction w.r.t. bending becomes possible
  // (may be necessary to accomodate a calo alignment)
  if (charge == 1)  // neg charge
    calophi = 20. / 180 * M_PI - calophi;
  calophi += 20. / 180 * M_PI;

  if (calophi < 0.) {
    edm::LogWarning("LUTMismatch")
        << "warning: calo offset goes into wrong direction by more than 20deg !!!! please correct!" << endl;
    calophi = 0.;
  }
  int cphi_ofs = (int)(calophi / (20. / 180. * M_PI));  // in 20 deg regions
  // 0; -1 region; 1 no offset; 2: +1 region , ... 7: +6 regions

  if (cphi_ofs > 7) {
    edm::LogWarning("LUTMismatch") << "warning: calo offset is larger than 6 regions !!!! please correct!" << endl;
    cphi_ofs = 7;
  }

  return ((cphi_fine << 3) + cphi_ofs);
}
