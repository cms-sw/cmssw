#include <assert.h>
#include <cstdlib>
#include "L1Trigger/L1TMuonEndCap/interface/PtLutVarCalc.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.hh"

PtAssignmentEngineAux2017 ENG;

int CalcTrackTheta( unsigned short theta, int st1_ring2, short mode ) {
    int nBits = (mode == 15 ? 4 : 5);
    theta = ENG.getTheta(theta, st1_ring2, nBits);
    return theta;
}

void CalcDeltaPhis( short& dPh12,   short& dPh13,    short& dPh14,   short& dPh23,    short& dPh24,   short& dPh34, short& dPhSign,
                    short& dPhSum4, short& dPhSum4A, short& dPhSum3, short& dPhSum3A, short& outStPh,
                    unsigned short mode ) {

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

    assert(dPhSign != 0);

    dPh12 *= dPhSign;
    dPh13 *= dPhSign;
    dPh14 *= dPhSign;
    dPh23 *= dPhSign;
    dPh24 *= dPhSign;
    dPh34 *= dPhSign;

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


  // Compute summed quantities
  if (mode == 15) {
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
  }

} // End function: CalcDeltaPhis()



void CalcDeltaThetas( short& dTh12, short& dTh13, short& dTh14, short& dTh23, short& dTh24, short& dTh34, short mode ) {
  
    int nBits = (mode == 15 ? 2 : 3);

    dTh12 = ENG.getdTheta(dTh12, nBits);
    dTh13 = ENG.getdTheta(dTh13, nBits);
    dTh14 = ENG.getdTheta(dTh14, nBits);
    dTh23 = ENG.getdTheta(dTh23, nBits);
    dTh24 = ENG.getdTheta(dTh24, nBits);
    dTh34 = ENG.getdTheta(dTh34, nBits);

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

    bend1 = ENG.getCLCT( pat1, endcap, dPhSign, nBits );
    bend2 = ENG.getCLCT( pat2, endcap, dPhSign, nBits );
    bend3 = ENG.getCLCT( pat3, endcap, dPhSign, nBits );
    bend4 = ENG.getCLCT( pat4, endcap, dPhSign, nBits );
  } // End conditional: if (BIT_COMP)

} // End function: CalcBends()

void CalcRPCs( int& RPC1, int& RPC2, int& RPC3, int& RPC4,
	       const int mode, const bool BIT_COMP ) {

  if (BIT_COMP) {
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

  assert( bend != -99 );
  return bend;
}

