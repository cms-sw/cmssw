#ifndef L1TMuonEndCap_PtAssignmentEngineAux_h
#define L1TMuonEndCap_PtAssignmentEngineAux_h

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

// This class (and its daughters) should never own any data members. It should have only functions.

class PtAssignmentEngineAux {
public:
  // Functions for GMT quantities
  int getGMTPt(float pt) const;
  int getGMTPtDxy(float pt) const;

  float getPtFromGMTPt(int gmt_pt) const;
  float getPtFromGMTPtDxy(int gmt_pt_dxy) const;

  int getGMTPhi(int phi) const;
  int getGMTPhiV2(int phi) const;

  int getGMTEta(int theta, int endcap) const;

  int getGMTQuality(int mode, int theta, bool promoteMode7, int version) const;

  std::pair<int, int> getGMTCharge(int mode, const std::vector<int>& phidiffs) const;

  int getGMTDxy(float dxy) const;
};

#endif
