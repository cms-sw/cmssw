#ifndef L1TMuonEndCap_PtAssignmentEngineAux2017_h
#define L1TMuonEndCap_PtAssignmentEngineAux2017_h

class PtAssignmentEngineAux2017 {
public:

  int getNLBdPhi(int dPhi, int bits=7, int max=512) const;

  int getNLBdPhiBin(int dPhi, int bits=7, int max=512) const;

  int getdPhiFromBin(int dPhiBin, int bits=7, int max=512) const;

  int getCLCT(int clct, int endcap, int dPhiSign, int bits=3) const;

  int unpackCLCT(int clct, int endcap, int dPhiSign, int bits) const;

  int getdTheta(int dTheta, int bits=3) const;

  int unpackdTheta(int dTheta, int bits) const;

  int getTheta(int theta, int ring2, int bits=5) const;

  void unpackTheta(int& theta, int& st1_ring2, int bits) const;

  int unpackSt1Ring2(int theta, int bits) const;

  int get2bRPC(int clctA, int clctB, int clctC) const;

  void unpack2bRPC(int rpc_2b, int& rpcA, int& rpcB, int& rpcC) const;

  int get8bMode15(int theta, int st1_ring2, int endcap, int sPhiAB,
                  int clctA, int clctB, int clctC, int clctD) const;

  void unpack8bMode15( int mode15_8b, int& theta, int& st1_ring2, int endcap, int sPhiAB,
                       int& clctA, int& rpcA, int& rpcB, int& rpcC, int& rpcD) const;

  // Need to re-check / verify this - AWB 17.03.17
  // int getFRLUT(int sector, int station, int chamber) const;
};

#endif

