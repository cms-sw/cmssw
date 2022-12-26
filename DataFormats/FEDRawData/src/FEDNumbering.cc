/** \file
 *
 *  \author G. Bruno  - CERN, EP Division
 */
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <cassert>

using namespace std;

namespace {

  constexpr std::array<bool, FEDNumbering::FEDNumbering::MAXFEDID + 1> initIn() {
    std::array<bool, FEDNumbering::MAXFEDID + 1> in = {};
    int i = 0;
    for (i = 0; i <= FEDNumbering::MAXFEDID; i++) {
      in[i] = false;
    }
    for (i = FEDNumbering::MINSiPixelFEDID; i <= FEDNumbering::MAXSiPixelFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINSiStripFEDID; i <= FEDNumbering::MAXSiStripFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINPreShowerFEDID; i <= FEDNumbering::MAXPreShowerFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINECALFEDID; i <= FEDNumbering::MAXECALFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCASTORFEDID; i <= FEDNumbering::MAXCASTORFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINLUMISCALERSFEDID; i <= FEDNumbering::MAXLUMISCALERSFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCSCFEDID; i <= FEDNumbering::MAXCSCFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCSCTFFEDID; i <= FEDNumbering::MAXCSCTFFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINDTFEDID; i <= FEDNumbering::MAXDTFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINDTTFFEDID; i <= FEDNumbering::MAXDTTFFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINRPCFEDID; i <= FEDNumbering::MAXRPCFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerGTPFEDID; i <= FEDNumbering::MAXTriggerGTPFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerEGTPFEDID; i <= FEDNumbering::MAXTriggerEGTPFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerGCTFEDID; i <= FEDNumbering::MAXTriggerGCTFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerLTCFEDID; i <= FEDNumbering::MAXTriggerLTCFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerLTCmtccFEDID; i <= FEDNumbering::MAXTriggerLTCmtccFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCSCDDUFEDID; i <= FEDNumbering::MAXCSCDDUFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCSCContingencyFEDID; i <= FEDNumbering::MAXCSCContingencyFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINCSCTFSPFEDID; i <= FEDNumbering::MAXCSCTFSPFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINDAQeFEDFEDID; i <= FEDNumbering::MAXDAQeFEDFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINDAQmFEDFEDID; i <= FEDNumbering::MAXDAQmFEDFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTCDSuTCAFEDID; i <= FEDNumbering::MAXTCDSuTCAFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINHCALuTCAFEDID; i <= FEDNumbering::MAXHCALuTCAFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINSiPixeluTCAFEDID; i <= FEDNumbering::MAXSiPixeluTCAFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINDTUROSFEDID; i <= FEDNumbering::MAXDTUROSFEDID; i++) {
      in[i] = true;
    }
    for (i = FEDNumbering::MINTriggerUpgradeFEDID; i <= FEDNumbering::MAXTriggerUpgradeFEDID; i++) {
      in[i] = true;
    }
    return in;
  }

  constexpr std::array<bool, FEDNumbering::MAXFEDID + 1> in_ = initIn();

}  // namespace

bool FEDNumbering::inRange(int i) { return in_[i]; }
bool FEDNumbering::inRangeNoGT(int i) {
  if ((i >= MINTriggerGTPFEDID && i <= MAXTriggerGTPFEDID) || (i >= MINTriggerEGTPFEDID && i <= MAXTriggerEGTPFEDID))
    return false;
  return in_[i];
}
