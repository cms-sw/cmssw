#ifndef L1TMuonEndCap_PtAssignmentEngineAux2017_h
#define L1TMuonEndCap_PtAssignmentEngineAux2017_h

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux.h"

class PtAssignmentEngineAux2017 : public PtAssignmentEngineAux {
public:
  int getNLBdPhi(int dPhi, int bits = 7, int max = 512) const;

  int getNLBdPhiBin(int dPhi, int bits = 7, int max = 512) const;

  int getdPhiFromBin(int dPhiBin, int bits = 7, int max = 512) const;

  int getCLCT(int clct, int endcap, int dPhiSign, int bits = 3) const;

  int unpackCLCT(int clct, int endcap, int dPhiSign, int bits) const;

  int getdTheta(int dTheta, int bits = 3) const;

  int unpackdTheta(int dTheta, int bits) const;

  int getTheta(int theta, int ring2, int bits = 5) const;

  void unpackTheta(int& theta, int& st1_ring2, int bits) const;

  int unpackSt1Ring2(int theta, int bits) const;

  int get2bRPC(int clctA, int clctB, int clctC) const;

  void unpack2bRPC(int rpc_2b, int& rpcA, int& rpcB, int& rpcC) const;

  int get8bMode15(int theta, int st1_ring2, int endcap, int sPhiAB, int clctA, int clctB, int clctC, int clctD) const;

  void unpack8bMode15(int mode15_8b,
                      int& theta,
                      int& st1_ring2,
                      int endcap,
                      int sPhiAB,
                      int& clctA,
                      int& rpcA,
                      int& rpcB,
                      int& rpcC,
                      int& rpcD) const;

  // Need to re-check / verify this - AWB 17.03.17
  // int getFRLUT(int sector, int station, int chamber) const;

  // ___________________________________________________________________________
  // From here down, code was originally in PtLUTVarCalc.h

  int calcTrackTheta(const int th1,
                     const int th2,
                     const int th3,
                     const int th4,
                     const int ring1,
                     const int mode,
                     const bool BIT_COMP = false) const;

  void calcDeltaPhis(int& dPh12,
                     int& dPh13,
                     int& dPh14,
                     int& dPh23,
                     int& dPh24,
                     int& dPh34,
                     int& dPhSign,
                     int& dPhSum4,
                     int& dPhSum4A,
                     int& dPhSum3,
                     int& dPhSum3A,
                     int& outStPh,
                     const int ph1,
                     const int ph2,
                     const int ph3,
                     const int ph4,
                     const int mode,
                     const bool BIT_COMP = false) const;

  void calcDeltaThetas(int& dTh12,
                       int& dTh13,
                       int& dTh14,
                       int& dTh23,
                       int& dTh24,
                       int& dTh34,
                       const int th1,
                       const int th2,
                       const int th3,
                       const int th4,
                       const int mode,
                       const bool BIT_COMP = false) const;

  void calcBends(int& bend1,
                 int& bend2,
                 int& bend3,
                 int& bend4,
                 const int pat1,
                 const int pat2,
                 const int pat3,
                 const int pat4,
                 const int dPhSign,
                 const int endcap,
                 const int mode,
                 const bool BIT_COMP = false) const;

  void calcRPCs(int& RPC1,
                int& RPC2,
                int& RPC3,
                int& RPC4,
                const int mode,
                const int st1_ring2,
                const int theta,
                const bool BIT_COMP = false) const;

  int calcBendFromPattern(const int pattern, const int endcap) const;

  void calcDeltaPhiSums(int& dPhSum4,
                        int& dPhSum4A,
                        int& dPhSum3,
                        int& dPhSum3A,
                        int& outStPh,
                        const int dPh12,
                        const int dPh13,
                        const int dPh14,
                        const int dPh23,
                        const int dPh24,
                        const int dPh34) const;
};

#endif
