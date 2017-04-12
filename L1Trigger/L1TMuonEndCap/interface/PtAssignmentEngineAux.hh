#ifndef L1TMuonEndCap_PtAssignmentEngineAux_hh
#define L1TMuonEndCap_PtAssignmentEngineAux_hh

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>


class PtAssignmentEngineAux {
public:
  // Functions for pT assignment
  const int (*getModeVariables() const)[6];

  int getNLBdPhi(int dPhi, int bits, int max=512) const;

  int getNLBdPhiBin(int dPhi, int bits, int max=512) const;

  int getdPhiFromBin(int dPhiBin, int bits, int max=512) const;

  int getCLCT(int clct) const;

  int getdTheta(int dTheta) const;

  int getdEta(int dEta) const;

  int getEtaInt(float eta, int bits=5) const;

  float getEtaFromThetaInt(int thetaInt, int bits=5) const;

  float getEtaFromEtaInt(int etaInt, int bits=5) const;

  float getEtaFromBin(int etaBin, int bits=5) const;

  int getFRLUT(int sector, int station, int chamber) const;

  // Functions for GMT quantities
  int getGMTPt(float pt) const;

  float getPtFromGMTPt(int gmt_pt) const;

  int getGMTPhi(int phi) const;
  int getGMTPhiV2(int phi) const;

  int getGMTEta(int theta, int endcap) const;

  int getGMTQuality(int mode, int theta) const;

  std::pair<int,int> getGMTCharge(int mode, const std::vector<int>& phidiffs) const;

};

#endif
