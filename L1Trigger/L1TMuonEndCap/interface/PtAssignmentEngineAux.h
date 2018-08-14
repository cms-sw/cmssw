#ifndef L1TMuonEndCap_PtAssignmentEngineAux_h
#define L1TMuonEndCap_PtAssignmentEngineAux_h

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>


class PtAssignmentEngineAux {
public:
  // Functions for GMT quantities
  int getGMTPt(float pt) const;

  float getPtFromGMTPt(int gmt_pt) const;

  int getGMTPhi(int phi) const;
  int getGMTPhiV2(int phi) const;

  int getGMTEta(int theta, int endcap) const;

  int getGMTQuality(int mode, int theta, bool promoteMode7, int version) const;

  std::pair<int,int> getGMTCharge(int mode, const std::vector<int>& phidiffs) const;

};

#endif
