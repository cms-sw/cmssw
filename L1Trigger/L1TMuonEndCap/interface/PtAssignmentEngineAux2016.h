#ifndef L1TMuonEndCap_PtAssignmentEngineAux2016_h
#define L1TMuonEndCap_PtAssignmentEngineAux2016_h

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>


class PtAssignmentEngineAux2016 {
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
};

#endif
