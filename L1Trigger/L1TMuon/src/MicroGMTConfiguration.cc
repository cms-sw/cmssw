#include "../interface/MicroGMTConfiguration.h"

unsigned
l1t::MicroGMTConfiguration::getTwosComp(const int signed_int, const int width) {
  if (signed_int >= 0) {
    return (unsigned)signed_int;
  }
  int all_one = (1 << width)-1;
  return ((-signed_int) ^ all_one) + 1;
}

int
l1t::MicroGMTConfiguration::calcGlobalPhi(int locPhi, tftype t, int proc) {
  int globPhi = 0;
  if (t == bmtf) {
      // each BMTF processor corresponds to a 30 degree wedge = 48 in int-scale
      globPhi = (proc) * 48 + locPhi;
      // first processor starts at CMS phi = -15 degrees...
      globPhi += 576-24;
      // handle wrap-around (since we add the 576-24, the value will never be negative!)
      globPhi = globPhi%576;
  } else {
      // all others correspond to 60 degree sectors = 96 in int-scale
      globPhi = (proc) * 96 + locPhi;
      // first processor starts at CMS phi = 15 degrees (24 in int)... Handle wrap-around with %. Add 576 to make sure the number is positive
      globPhi = (globPhi + 600) % 576;
  }
  return globPhi;
}

int
l1t::MicroGMTConfiguration::setOutputMuonQuality(int muQual, tftype type, int haloBit) {
  // use only the two MSBs for the muon to the uGT
  int outQual = muQual & 0xC;
  if (haloBit == 1 && (type == tftype::emtf_neg || type == tftype::emtf_pos)) {
    // set quality to 0xF if the halo bit is set
    outQual = 0xF;
  }
  return outQual;
}
