#ifndef L1TMuonEndCap_PtAssignmentEngineAux2017_hh
#define L1TMuonEndCap_PtAssignmentEngineAux2017_hh

class PtAssignmentEngineAux2017 {
public:
  // // Functions for pT assignment
  // const int (*getModeVariables() const)[6];

  int getNLBdPhi(int dPhi, int bits=7, int max=512) const;

  int getNLBdPhiBin(int dPhi, int bits=7, int max=512) const;

  int getdPhiFromBin(int dPhiBin, int bits=7, int max=512) const;

  int getCLCT(int clct, int endcap, int dPhiSign, int bits=3) const;

  int getdTheta(int dTheta, int bits=3) const;

  int getTheta(int theta, int ring2, int bits=5) const;

  // Need to re-check / verify this - AWB 17.03.17
  // int getFRLUT(int sector, int station, int chamber) const;
};

#endif

