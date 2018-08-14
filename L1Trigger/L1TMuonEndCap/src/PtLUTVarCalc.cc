#include <cassert>
#include <cstdlib>
#include "L1Trigger/L1TMuonEndCap/interface/PtLUTVarCalc.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// From here down, exact copy of code used for training BDT: EMTFPtAssign2017/src/PtLUTVarCalc.cc

PtAssignmentEngineAux2017 ENG;


int CalcTrackTheta( const int th1, const int th2, const int th3, const int th4,
                    const int st1_ring2, const int mode, const bool BIT_COMP ) {

  int theta = -99;

  if      ( (mode % 8) / 4 > 0 ) // Has station 2 hit
    theta = th2;
  else if ( (mode % 4) / 2 > 0 ) // Has station 3 hit
    theta = th3;
  else if ( (mode % 2) > 0 ) // Has station 4 hit
    theta = th4;

  if (not( theta > 0 ))
    { edm::LogError("L1T") << "theta = " << theta; return 0; }

  if (BIT_COMP) {
    int nBits = (mode == 15 ? 4 : 5);
    theta = ENG.getTheta(theta, st1_ring2, nBits);
  }

  return theta;
}


void CalcDeltaPhis( int& dPh12, int& dPh13, int& dPh14, int& dPh23, int& dPh24, int& dPh34, int& dPhSign,
                    int& dPhSum4, int& dPhSum4A, int& dPhSum3, int& dPhSum3A, int& outStPh,
                    const int ph1, const int ph2, const int ph3, const int ph4, const int mode, const bool BIT_COMP ) {

  dPh12 = ph2 - ph1;
  dPh13 = ph3 - ph1;
  dPh14 = ph4 - ph1;
  dPh23 = ph3 - ph2;
  dPh24 = ph4 - ph2;
  dPh34 = ph4 - ph3;
  dPhSign = 0;

  if (mode >= 8) {                   // First hit is station 1
    if      ( (mode % 8) / 4 > 0 )   // Has station 2 hit
      dPhSign = (dPh12 >= 0 ? +1 : -1);
    else if ( (mode % 4) / 2 > 0 )   // Has station 3 hit
      dPhSign = (dPh13 >= 0 ? +1 : -1);
    else if ( (mode % 2) > 0 )       // Has station 4 hit
      dPhSign = (dPh14 >= 0 ? +1 : -1);
  } else if ( (mode % 8) / 4 > 0 ) { // First hit is station 2
    if      ( (mode % 4) / 2 > 0 )   // Has station 3 hit
      dPhSign = (dPh23 >= 0 ? +1 : -1);
    else if ( (mode % 2) > 0 )       // Has station 4 hit
      dPhSign = (dPh24 >= 0 ? +1 : -1);
  } else if ( (mode % 4) / 2 > 0 ) { // First hit is station 3
    if      ( (mode % 2) > 0 )       // Has station 4 hit
      dPhSign = (dPh34 >= 0 ? +1 : -1);
  }

  if (not(dPhSign != 0))
    { edm::LogError("L1T") << "dPhSign = " << dPhSign; return; }

  dPh12 *= dPhSign;
  dPh13 *= dPhSign;
  dPh14 *= dPhSign;
  dPh23 *= dPhSign;
  dPh24 *= dPhSign;
  dPh34 *= dPhSign;

  if (BIT_COMP) {
    int nBitsA = 7;
    int nBitsB = 7;
    int nBitsC = 7;
    int maxA = 512;
    int maxB = 512;
    int maxC = 512;

    if (mode == 7 || mode == 11 || mode > 12) {
      nBitsB = 5;
      maxB = 256;
      nBitsC = 5;
      maxC = 256;
    }
    if (mode == 15) {
      nBitsC = 4;
      maxC = 256;
    }

    dPh12 = ENG.getNLBdPhi(dPh12, nBitsA, maxA);
    dPh13 = ENG.getNLBdPhi(dPh13, nBitsA, maxA);
    dPh14 = ENG.getNLBdPhi(dPh14, nBitsA, maxA);
    if (mode == 7)
      dPh23 = ENG.getNLBdPhi(dPh23, nBitsA, maxA);
    else
      dPh23 = ENG.getNLBdPhi(dPh23, nBitsB, maxB);
    dPh24 = ENG.getNLBdPhi(dPh24, nBitsB, maxB);
    dPh34 = ENG.getNLBdPhi(dPh34, nBitsC, maxC);

    // Some delta phi values must be computed from others
    switch (mode) {
    case 15:  dPh13 = dPh12 + dPh23;  dPh14 = dPh13 + dPh34;  dPh24 = dPh23 + dPh34;  break;
    case 14:  dPh13 = dPh12 + dPh23;  break;
    case 13:  dPh14 = dPh12 + dPh24;  break;
    case 11:  dPh14 = dPh13 + dPh34;  break;
    case  7:  dPh24 = dPh23 + dPh34;  break;
    default:  break;
    }

  } // End conditional: if (BIT_COMP)


  // Compute summed quantities
  if (mode == 15) CalcDeltaPhiSums( dPhSum4, dPhSum4A, dPhSum3, dPhSum3A, outStPh,
                                    dPh12,  dPh13,  dPh14,  dPh23,  dPh24,  dPh34 );

} // End function: CalcDeltaPhis()



void CalcDeltaThetas( int& dTh12, int& dTh13, int& dTh14, int& dTh23, int& dTh24, int& dTh34,
                      const int th1, const int th2, const int th3, const int th4, const int mode, const bool BIT_COMP ) {

  dTh12 = th2 - th1;
  dTh13 = th3 - th1;
  dTh14 = th4 - th1;
  dTh23 = th3 - th2;
  dTh24 = th4 - th2;
  dTh34 = th4 - th3;

  if (BIT_COMP) {
    int nBits = (mode == 15 ? 2 : 3);

    dTh12 = ENG.getdTheta(dTh12, nBits);
    dTh13 = ENG.getdTheta(dTh13, nBits);
    dTh14 = ENG.getdTheta(dTh14, nBits);
    dTh23 = ENG.getdTheta(dTh23, nBits);
    dTh24 = ENG.getdTheta(dTh24, nBits);
    dTh34 = ENG.getdTheta(dTh34, nBits);
  } // End conditional: if (BIT_COMP)

} // CalcDeltaThetas()



void CalcBends( int& bend1, int& bend2, int& bend3, int& bend4,
                const int pat1, const int pat2, const int pat3, const int pat4,
                const int dPhSign, const int endcap, const int mode, const bool BIT_COMP ) {

  bend1 = CalcBendFromPattern( pat1, endcap );
  bend2 = CalcBendFromPattern( pat2, endcap );
  bend3 = CalcBendFromPattern( pat3, endcap );
  bend4 = CalcBendFromPattern( pat4, endcap );

  if (BIT_COMP) {
    int nBits = 3;
    if (mode == 7 || mode == 11 || mode > 12)
      nBits = 2;

    if (  mode      / 8 > 0 ) // Has station 1 hit
      bend1 = ENG.getCLCT( pat1, endcap, dPhSign, nBits );
    if ( (mode % 8) / 4 > 0 ) // Has station 2 hit
      bend2 = ENG.getCLCT( pat2, endcap, dPhSign, nBits );
    if ( (mode % 4) / 2 > 0 ) // Has station 3 hit
      bend3 = ENG.getCLCT( pat3, endcap, dPhSign, nBits );
    if ( (mode % 2)     > 0 ) // Has station 4 hit
      bend4 = ENG.getCLCT( pat4, endcap, dPhSign, nBits );
  } // End conditional: if (BIT_COMP)

} // End function: CalcBends()

void CalcRPCs( int& RPC1, int& RPC2, int& RPC3, int& RPC4, const int mode,
               const int st1_ring2, const int theta, const bool BIT_COMP ) {

  if (BIT_COMP) {

    // Mask some invalid locations for RPC hits
    // theta is assumed to be the compressed, mode 15 version
    if (mode == 15 && !st1_ring2) {
      RPC1 = 0;
      RPC2 = 0;
      if (theta < 4) {
        RPC3 = 0;
        RPC4 = 0;
      }
    }

    int nRPC = (RPC1 == 1) + (RPC2 == 1) + (RPC3 == 1) + (RPC4 == 1);

    // In 3- and 4-station modes, only specify some combinations of RPCs
    if (nRPC >= 2) {

      if        (mode == 15) {
        if        (RPC1 == 1 && RPC2 == 1) {
          RPC3 = 0;
          RPC4 = 0;
        } else if (RPC1 == 1 && RPC3 == 1) {
          RPC4 = 0;
        } else if (RPC4 == 1 && RPC2 == 1) {
          RPC3 = 0;
        } else if (RPC3 == 1 && RPC4 == 1 && !st1_ring2) {
          RPC3 = 0;
        }
      } else if (mode == 14) {
        if        (RPC1 == 1) {
          RPC2 = 0;
          RPC3 = 0;
        } else if (RPC3 == 1) {
          RPC2 = 0;
        }
      } else if (mode == 13) {
        if        (RPC1 == 1) {
          RPC2 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC2 = 0;
        }
      } else if (mode == 11) {
        if        (RPC1 == 1) {
          RPC3 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC3 = 0;
        }
      } else if (mode == 7) {
        if        (RPC2 == 1) {
          RPC3 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC3 = 0;
        }
      }

    } // End conditional: if (nRPC >= 2)
  } // End conditional: if (BIT_COMP)

} // End function: void CalcRPCs()


int CalcBendFromPattern( const int pattern, const int endcap ) {

  int bend = -99;
  if (pattern < 0)
    return bend;

  if (pattern == 10)
    bend = 0;
  else if ( (pattern % 2) == 0 )
    bend = (10 - pattern) / 2;
  else if ( (pattern % 2) == 1 )
    bend = -1 * (11 - pattern) / 2;

  // Reverse to match dPhi convention
  if (endcap == 1)
    bend *= -1;

  if (not( bend != -99 ))
    { edm::LogError("L1T") << "bend = " << bend; return 0; }
  return bend;
}


void CalcDeltaPhiSums( int& dPhSum4, int& dPhSum4A, int& dPhSum3, int& dPhSum3A, int& outStPh,
                       const int dPh12, const int dPh13, const int dPh14, const int dPh23, const int dPh24, const int dPh34 ) {

    dPhSum4  = dPh12 + dPh13 + dPh14 + dPh23 + dPh24 + dPh34;
    dPhSum4A = abs(dPh12) + abs(dPh13) + abs(dPh14) + abs(dPh23) + abs(dPh24) + abs(dPh34);
    int devSt1 = abs(dPh12) + abs(dPh13) + abs(dPh14);
    int devSt2 = abs(dPh12) + abs(dPh23) + abs(dPh24);
    int devSt3 = abs(dPh13) + abs(dPh23) + abs(dPh34);
    int devSt4 = abs(dPh14) + abs(dPh24) + abs(dPh34);

    if      (devSt4 > devSt3 && devSt4 > devSt2 && devSt4 > devSt1)  outStPh = 4;
    else if (devSt3 > devSt4 && devSt3 > devSt2 && devSt3 > devSt1)  outStPh = 3;
    else if (devSt2 > devSt4 && devSt2 > devSt3 && devSt2 > devSt1)  outStPh = 2;
    else if (devSt1 > devSt4 && devSt1 > devSt3 && devSt1 > devSt2)  outStPh = 1;
    else                                                             outStPh = 0;

    if      (outStPh == 4) {
      dPhSum3  = dPh12 + dPh13 + dPh23;
      dPhSum3A = abs(dPh12) + abs(dPh13) + abs(dPh23);
    } else if (outStPh == 3) {
      dPhSum3  = dPh12 + dPh14 + dPh24;
      dPhSum3A = abs(dPh12) + abs(dPh14) + abs(dPh24);
    } else if (outStPh == 2) {
      dPhSum3  = dPh13 + dPh14 + dPh34;
      dPhSum3A = abs(dPh13) + abs(dPh14) + abs(dPh34);
    } else {
      dPhSum3  = dPh23 + dPh24 + dPh34;
      dPhSum3A = abs(dPh23) + abs(dPh24) + abs(dPh34);
    }

} // End function: void CalcDeltaPhiSums()
