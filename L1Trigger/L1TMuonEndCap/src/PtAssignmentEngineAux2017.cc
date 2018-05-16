#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream> 
#include <cassert>
#include <cmath>

// From here down, exact copy of code used for training BDT: EMTFPtAssign2017/src/PtLUTVarCalc.cc


// Arrays that map the integer dPhi --> dPhi-units. 1/60th of a degree per unit; 255 units --> 4.25 degrees, 511 --> 8.52 degrees

// 256 max units----
// For use in dPhi34 in mode 15.  Derived manually from dPhiNLBMap_5bit_256Max for now; should generate algorithmically. - AWB 17.03.17
static const int dPhiNLBMap_4bit_256Max[16] = {0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 31, 46, 68, 136};

// For use in dPhi23, dPhi24, and dPhi34 in 3- and 4-station modes (7, 11, 13, 14, 15), except for dPhi23 in mode 7 and dPhi34 in mode 15
static const int dPhiNLBMap_5bit_256Max[32] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                                               16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136};
// 512 max units----
// For use in all dPhiAB (where "A" and "B" are the first two stations in the track) in all modes
static const int dPhiNLBMap_7bit_512Max[128] =  {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
                                                  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
                                                  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
                                                  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
                                                  64,  65,  66,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  79,  80,  81,
                                                  83,  84,  86,  87,  89,  91,  92,  94,  96,  98, 100, 102, 105, 107, 110, 112,
                                                 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168, 174, 181,
                                                 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470};


int PtAssignmentEngineAux2017::getNLBdPhi(int dPhi, int bits, int max) const {
  if (not( (bits == 4 && max == 256) || 
	   (bits == 5 && max == 256) || 
	   (bits == 7 && max == 512) ))
    { edm::LogError("L1T") << "bits = " << bits << ", max = " << max; return 0; }

  int dPhi_ = max;
  int sign_ = 1;
  if (dPhi < 0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max == 256) {
    if (bits == 4) {
      dPhi_ = dPhiNLBMap_4bit_256Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_4bit_256Max[edge]  <= dPhi &&
            dPhiNLBMap_4bit_256Max[edge+1] > dPhi) {
          dPhi_ = dPhiNLBMap_4bit_256Max[edge];
          break;
        }
      }
    } // End conditional: if (bits == 4)
    if (bits == 5) {
      dPhi_ = dPhiNLBMap_5bit_256Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_5bit_256Max[edge]  <= dPhi &&
            dPhiNLBMap_5bit_256Max[edge+1] > dPhi) {
          dPhi_ = dPhiNLBMap_5bit_256Max[edge];
          break;
        }
      }
    } // End conditional: if (bits == 5)
  } // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7) {
      dPhi_ = dPhiNLBMap_7bit_512Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_7bit_512Max[edge]  <= dPhi &&
            dPhiNLBMap_7bit_512Max[edge+1] > dPhi) {
          dPhi_ = dPhiNLBMap_7bit_512Max[edge];
          break;
        }
      }
    } // End conditional: if (bits == 7)
  } // End conditional: else if (max == 512)

  if (not( abs(sign_) == 1 && dPhi_ >= 0 && dPhi_ < max))
    { edm::LogError("L1T") << "sign_ = " << sign_ << ", dPhi_ = " << dPhi_ << ", max = " << max; return 0; }
  return (sign_ * dPhi_);
} // End function: nt PtAssignmentEngineAux2017::getNLBdPhi()


int PtAssignmentEngineAux2017::getNLBdPhiBin(int dPhi, int bits, int max) const {
  if (not( (bits == 4 && max == 256) || 
	   (bits == 5 && max == 256) || 
	   (bits == 7 && max == 512) ))
    { edm::LogError("L1T") << "bits = " << bits << ", max = " << max; return 0; }
  
  int dPhiBin_ = (1 << bits) - 1;
  int sign_ = 1;
  if (dPhi < 0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max == 256) {
    if (bits == 4) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_4bit_256Max[edge] <= dPhi &&
            dPhiNLBMap_4bit_256Max[edge+1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    } // End conditional: if (bits == 4)
    if (bits == 5) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_5bit_256Max[edge]  <= dPhi &&
            dPhiNLBMap_5bit_256Max[edge+1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    } // End conditional: if (bits == 5)
  } // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_7bit_512Max[edge]  <= dPhi &&
            dPhiNLBMap_7bit_512Max[edge+1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    } // End conditional: if (bits == 7)
  } // End conditional: else if (max == 512)

  if (not(dPhiBin_ >= 0 && dPhiBin_ < pow(2, bits)))
    { edm::LogError("L1T") << "dPhiBin_ = " << dPhiBin_ << ", bits = " << bits; return 0; }
  return (dPhiBin_);
} // End function: int PtAssignmentEngineAux2017::getNLBdPhiBin()


int PtAssignmentEngineAux2017::getdPhiFromBin(int dPhiBin, int bits, int max) const {
  if (not( (bits == 4 && max == 256) || 
	   (bits == 5 && max == 256) || 
	   (bits == 7 && max == 512) ))
    { edm::LogError("L1T") << "bits = " << bits << ", max = " << max; return 0; }
  
  int dPhi_ = (1 << bits) - 1;

  if (dPhiBin > (1 << bits) - 1)
    dPhiBin = (1 << bits) - 1;

  if (max == 256) {
    if (bits == 4)
      dPhi_ = dPhiNLBMap_4bit_256Max[dPhiBin];
    if (bits == 5)
      dPhi_ = dPhiNLBMap_5bit_256Max[dPhiBin];
  } // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7)
      dPhi_ = dPhiNLBMap_7bit_512Max[dPhiBin];
  } // End conditional: else if (max == 512)

  if (not(dPhi_ >= 0 && dPhi_ < max))
    { edm::LogError("L1T") << "dPhi_ = " << dPhi_ << ", max = " << max; return 0; }
  return (dPhi_);
} // End function: int PtAssignmentEngineAux2017::getdPhiFromBin()


int PtAssignmentEngineAux2017::getCLCT(int clct, int endcap, int dPhiSign, int bits) const {

  // std::cout << "Inside getCLCT: clct = " << clct << ", endcap = " << endcap
  //             << ", dPhiSign = " << dPhiSign << ", bits = " << bits << std::endl;

  if (not( clct >= 0 && clct <= 10 && abs(endcap) == 1 && 
	   abs(dPhiSign) == 1 && (bits == 2 || bits == 3) ))
    { edm::LogError("L1T") << "clct = " << clct << ", endcap = " << endcap
			   << ", dPhiSign = " << dPhiSign << ", bits = " << bits; return 0; }

  // Convention here: endcap == +/-1, dPhiSign = +/-1.
  int clct_ = 0;
  int sign_ = -1 * endcap * dPhiSign;  // CLCT bend is with dPhi in ME-, opposite in ME+

  // CLCT pattern can be converted into |bend| x sign as follows:
  // |bend| = (10 + (pattern % 2) - pattern) / 2
  //   * 10 --> 0, 9/8 --> 1, 7/6 --> 2, 5/4 --> 3, 3/2 --> 4, 0 indicates RPC hit
  //  sign  = ((pattern % 2) == 1 ? -1 : 1) * (endcap == 1 ? -1 : 1)
  //   * In ME+, even CLCTs have negative sign, odd CLCTs have positive

  // For use in all 3- and 4-station modes (7, 11, 13, 14, 15)
  // Bends [-4, -3, -2] --> 0, [-1, 0] --> 1, [+1] --> 2, [+2, +3, +4] --> 3
  if (bits == 2) {
    switch (clct) {
    case 10: clct_ = 1;                   break;
    case  9: clct_ = (sign_ > 0 ? 1 : 2); break;
    case  8: clct_ = (sign_ > 0 ? 2 : 1); break;
    case  7: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  6: clct_ = (sign_ > 0 ? 3 : 0); break;
    case  5: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  4: clct_ = (sign_ > 0 ? 3 : 0); break;
    case  3: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  2: clct_ = (sign_ > 0 ? 3 : 0); break;
    case  1: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  0: clct_ = 0;                   break;
    default: clct_ = 1;                   break;
    }
  } // End conditional: if (bits == 2)

  // For use in all 2-station modes (3, 5, 6, 9, 10, 12)
  // Bends [isRPC] --> 0, [-4, -3] --> 1, [-2] --> 2, [-1] --> 3, [0] --> 4, [+1] --> 5, [+2] --> 6, [+3, +4] --> 7
  else if (bits == 3) {
    switch (clct) {
    case 10: clct_ = 4;                   break;
    case  9: clct_ = (sign_ > 0 ? 3 : 5); break;
    case  8: clct_ = (sign_ > 0 ? 5 : 3); break;
    case  7: clct_ = (sign_ > 0 ? 2 : 6); break;
    case  6: clct_ = (sign_ > 0 ? 6 : 2); break;
    case  5: clct_ = (sign_ > 0 ? 1 : 7); break;
    case  4: clct_ = (sign_ > 0 ? 7 : 1); break;
    case  3: clct_ = (sign_ > 0 ? 1 : 7); break;
    case  2: clct_ = (sign_ > 0 ? 7 : 1); break;
    case  1: clct_ = (sign_ > 0 ? 1 : 7); break;
    case  0: clct_ = 0;                   break;
    default: clct_ = 4;                   break;
    }
  } // End conditional: else if (bits == 3)

  // std::cout << "  * Output clct_ = " << clct_ << std::endl;

  if (not(clct_ >= 0 && clct_ < pow(2, bits)))
    { edm::LogError("L1T") << "clct_ = " << clct_ << ", bits = " << bits; return 0; }
  return clct_;
} // End function: int PtAssignmentEngineAux2017::getCLCT()


int PtAssignmentEngineAux2017::unpackCLCT(int clct, int endcap, int dPhiSign, int bits) const {

  // std::cout << "Inside unpackCLCT: clct = " << clct << ", endcap = " << endcap
  //             << ", dPhiSign = " << dPhiSign << ", bits = " << bits << std::endl;

  if (not(bits == 2 || bits == 3))
  { edm::LogError("L1T") << "bits = " << bits; return 0; }
  if (not(clct >= 0 && clct < pow(2, bits)))
    { edm::LogError("L1T") << "bits = " << bits << ", clct = " << clct; return 0; }
  if (not(abs(dPhiSign) == 1))
    { edm::LogError("L1T") << "dPhiSign = " << dPhiSign; return 0; }

  // Convention here: endcap == +/-1, dPhiSign = +/-1.
  int clct_ = -1;
  int sign_ = -1 * endcap * dPhiSign;  // CLCT bend is with dPhi in ME-, opposite in ME+

  if (bits == 2) {
    switch (clct) {
    case 1: clct_ = 10;                  break;
    case 2: clct_ = (sign_ > 0 ? 8 : 9); break;
    case 3: clct_ = (sign_ > 0 ? 4 : 5); break;
    case 0: clct_ =  0;                  break;
    default: break;
    }
  } else if (bits == 3) {
    switch (clct) {
    case 4: clct_ = 10;                  break;
    case 5: clct_ = (sign_ > 0 ? 8 : 9); break;
    case 3: clct_ = (sign_ > 0 ? 9 : 8); break;
    case 6: clct_ = (sign_ > 0 ? 6 : 7); break;
    case 2: clct_ = (sign_ > 0 ? 7 : 6); break;
    case 7: clct_ = (sign_ > 0 ? 4 : 5); break;
    case 1: clct_ = (sign_ > 0 ? 5 : 4); break;
    case 0: clct_ =  0;                  break;
    default: break;
    }
  }

  // std::cout << "  * Output clct_ = " << clct_ << std::endl;

  if (not(clct_ >= 0 && clct_ <= 10))
    { edm::LogError("L1T") << "clct_ = " << clct_; return 0; }
  return clct_;
} // End function: int PtAssignmentEngineAux2017::unpackCLCT()


int PtAssignmentEngineAux2017::getdTheta(int dTheta, int bits) const {
  if (not( bits == 2 || bits == 3 ))
    { edm::LogError("L1T") << "bits = " << bits; return 0; }

  int dTheta_ = -99;

  // For use in mode 15
  if (bits == 2) {
    if      (abs(dTheta) <= 1)
      dTheta_ = 2;
    else if (abs(dTheta) <= 2)
      dTheta_ = 1;
    else if (dTheta <= -3)
      dTheta_ = 0;
    else
      dTheta_ = 3;
  } // End conditional: if (bits == 2)

  // For use in all 2- and 3-station modes (all modes except 15)
  else if (bits == 3) {
    if      (dTheta <= -4)
      dTheta_ = 0;
    else if (dTheta == -3)
      dTheta_ = 1;
    else if (dTheta == -2)
      dTheta_ = 2;
    else if (dTheta == -1)
      dTheta_ = 3;
    else if (dTheta ==  0)
      dTheta_ = 4;
    else if (dTheta == +1)
      dTheta_ = 5;
    else if (dTheta == +2)
      dTheta_ = 6;
    else
      dTheta_ = 7;
  } // End conditional: if (bits == 3)

  if (not(dTheta_ >= 0 && dTheta_ < pow(2, bits)))
    { edm::LogError("L1T") << "dTheta_ = " << dTheta_ << ", bits = " << bits; return 0; }
  return (dTheta_);
} // End function: int PtAssignmentEngineAux2017::getdTheta()


int PtAssignmentEngineAux2017::unpackdTheta(int dTheta, int bits) const {
  if (not( bits == 2 || bits == 3 ))
    { edm::LogError("L1T") << "bits = " << bits; return 0; }
  int dTheta_ = -99;

  if        (bits == 2) { // For use in mode 15
    switch (dTheta) {
    case 2: dTheta_ =  0; break;
    case 1: dTheta_ = -2; break;
    case 0: dTheta_ = -3; break;
    case 3: dTheta_ =  3; break;
    default: break;
    }
  } else if (bits == 3) { // For use in all 2- and 3-station modes (all modes except 15)
    switch (dTheta) {
    case 0: dTheta_ = -4; break;
    case 1: dTheta_ = -3; break;
    case 2: dTheta_ = -2; break;
    case 3: dTheta_ = -1; break;
    case 4: dTheta_ =  0; break;
    case 5: dTheta_ =  1; break;
    case 6: dTheta_ =  2; break;
    case 7: dTheta_ =  3; break;
    default: break;
    }
  }

  if (not(dTheta_ >= -4 && dTheta_ <= 3))
    { edm::LogError("L1T") << "dTheta_ = " << dTheta_; return 0; }
  return (dTheta_);
} // End function: int PtAssignmentEngineAux2017::unpackdTheta(int dTheta, int bits)


int PtAssignmentEngineAux2017::getTheta(int theta, int st1_ring2, int bits) const {
  if (not( theta >= 5 && theta < 128 && 
	   (st1_ring2 == 0 || st1_ring2 == 1) && 
	   (bits == 4 || bits == 5) ))
    { edm::LogError("L1T") << "theta = " << theta << ", st1_ring2 = " << st1_ring2
			   << ", bits = " << bits; return 0; }

  int theta_ = -99;

  // For use in mode 15
  if (bits == 4) {
    if (st1_ring2 == 0) {
      // Should rarely fail ... should change to using ME1 for track theta - AWB 05.06.17
      if (theta > 52) {
        // std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/1 LCT and track theta = " << theta << std::endl;
      }
      theta_ = (std::min( std::max(theta, 5), 52) - 5) / 6;
    }
    else if (st1_ring2 == 1) {
      // Should rarely fail ... should change to using ME1 for track theta - AWB 05.06.17
      if (theta < 46 || theta > 87) {
        // std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/2 LCT and track theta = " << theta << std::endl;
      }
      theta_ = ((std::min( std::max(theta, 46), 87) - 46) / 7) + 8;
    }
  } // End conditional: if (bits == 4)

  // For use in all 2- and 3-station modes (all modes except 15)
  else if (bits == 5) {
    if (st1_ring2 == 0) {
      theta_ = (std::max(theta, 1) - 1) / 4;
    }
    else if (st1_ring2 == 1) {
      theta_ = ((std::min(theta, 104) - 1) / 4) + 6;
    }
  } // End conditional: else if (bits == 5)

  if (not(theta_ >= 0 && ((bits == 4 && theta_ <= 13) || (bits == 5 && theta_ < pow(2, bits))) ))
    { edm::LogError("L1T") << "theta_ = " << theta_ << ", bits = " << bits; return 0; }
  return (theta_);
} // End function: int PtAssignmentEngineAux2017::getTheta()


void PtAssignmentEngineAux2017::unpackTheta(int& theta, int& st1_ring2, int bits) const {
  if (not(bits == 4 || bits == 5))
    { edm::LogError("L1T") << "bits = " << bits; return; }
  if (not(theta >= 0 && theta < pow(2, bits)))
    { edm::LogError("L1T") << "theta = " << theta << ", bits = " << bits; return; }

  // For use in mode 15
  if (bits == 4) {
    if (theta < 8) {
      st1_ring2 = 0;
      theta = (theta * 6) + 5;
    } else {
      st1_ring2 = 1;
      theta = ((theta - 8) * 7) + 46;
    }
  } else if (bits == 5) {
    if (theta < 15) {
      st1_ring2 = 0;
      theta = (theta * 4) + 1;
    } else {
      st1_ring2 = 1;
      theta = ((theta - 6) * 4) + 1;
    }
  }

  if (not(theta >= 5 && theta <= 104))
    { edm::LogError("L1T") << "theta = " << theta; return; }

} // End function: void PtAssignmentEngineAux2017::unpackTheta()


int PtAssignmentEngineAux2017::unpackSt1Ring2(int theta, int bits) const {
  if (not(bits == 4 || bits == 5))
    { edm::LogError("L1T") << "bits = " << bits; return 0; }
  if (not(theta >= 0 && theta < pow(2, bits)))
    { edm::LogError("L1T") << "theta = " << theta << ", bits = " << bits; return 0; }

  // For use in mode 15
  if (bits == 4) {
    if (theta < 6) return 0;
    else           return 1;
  } else {
    if (theta < 15) return 0;
    else            return 1;
  }

} // End function: void PtAssignmentEngineAux2017::unpackSt1Ring2()


int PtAssignmentEngineAux2017::get2bRPC(int clctA, int clctB, int clctC) const {

  int rpc_2b = -99;

  if      (clctA == 0) rpc_2b = 0;
  else if (clctC == 0) rpc_2b = 1;
  else if (clctB == 0) rpc_2b = 2;
  else                 rpc_2b = 3;

  if (not(rpc_2b >= 0 && rpc_2b < 4))
  { edm::LogError("L1T") << "rpc_2b = " << rpc_2b; return 0; }
  return (rpc_2b);
} // End function: int PtAssignmentEngineAux2017::get2bRPC()


void PtAssignmentEngineAux2017::unpack2bRPC(int rpc_2b, int& rpcA, int& rpcB, int& rpcC) const {

  if (not(rpc_2b >= 0 && rpc_2b < 4))
  { edm::LogError("L1T") << "rpc_2b = " << rpc_2b; return; }
  
  rpcA = 0; rpcB = 0; rpcC = 0;

  if      (rpc_2b == 0) rpcA = 1;
  else if (rpc_2b == 1) rpcC = 1;
  else if (rpc_2b == 2) rpcB = 1;

} // End function: int PtAssignmentEngineAux2017::unpack2bRPC()


int PtAssignmentEngineAux2017::get8bMode15(int theta, int st1_ring2, int endcap, int sPhiAB,
                                           int clctA, int clctB, int clctC, int clctD) const {

  // std::cout << "Inside get8bMode15, theta = " << theta << ", st1_ring2 = " << st1_ring2 << ", endcap = " << endcap << ", sPhiAB = " << sPhiAB
  //             << ", clctA = " << clctA << ", clctB = " << clctB << ", clctC = " << clctC << ", clctD = " << clctD << std::endl;

  if (st1_ring2) theta = (std::min( std::max(theta, 46), 87) - 46) / 7;
  else           theta = (std::min( std::max(theta,  5), 52) -  5) / 6;
  if (not(theta >= 0 && theta < 10))
    { edm::LogError("L1T") << "theta = " << theta; return 0; }
  
  int clctA_2b = getCLCT(clctA, endcap, sPhiAB, 2);

  int nRPC = (clctA == 0) + (clctB == 0) + (clctC == 0) + (clctD == 0);
  int rpc_word, rpc_clct, mode15_8b;

  if (st1_ring2) {
    if      (nRPC >= 2 && clctA == 0 && clctB == 0) rpc_word =  0;
    else if (nRPC >= 2 && clctA == 0 && clctC == 0) rpc_word =  1;
    else if (nRPC >= 2 && clctA == 0 && clctD == 0) rpc_word =  2;
    else if (nRPC == 1 && clctA == 0              ) rpc_word =  3;
    else if (nRPC >= 2 && clctD == 0 && clctB == 0) rpc_word =  4;
    else if (nRPC >= 2 && clctD == 0 && clctC == 0) rpc_word =  8;
    else if (nRPC >= 2 && clctB == 0 && clctC == 0) rpc_word = 12;
    else if (nRPC == 1 && clctD == 0              ) rpc_word = 16;
    else if (nRPC == 1 && clctB == 0              ) rpc_word = 20;
    else if (nRPC == 1 && clctC == 0              ) rpc_word = 24;
    else                                            rpc_word = 28;
    rpc_clct  = rpc_word + clctA_2b;
    mode15_8b = (theta*32) + rpc_clct + 64;
  } else {
    if      (theta >= 4 && clctD == 0) rpc_word = 0;
    else if (theta >= 4 && clctC == 0) rpc_word = 1;
    else if (theta >= 4              ) rpc_word = 2;
    else                               rpc_word = 3;
    rpc_clct  = rpc_word*4 + clctA_2b;
    mode15_8b = ((theta % 4)*16) + rpc_clct;
  }

  // std::cout << "  * Output mode15_8b = " << mode15_8b << std::endl;

  if (not(mode15_8b >= 0 && mode15_8b < pow(2, 8)))
    { edm::LogError("L1T") << "mode15_8b = " << mode15_8b; return 0; }
  return (mode15_8b);

} // End function: int PtAssignmentEngineAux2017::get8bMode15()


void PtAssignmentEngineAux2017::unpack8bMode15( int mode15_8b, int& theta, int& st1_ring2, int endcap, int sPhiAB,
                                                int& clctA, int& rpcA, int& rpcB, int& rpcC, int& rpcD) const {

  // std::cout << "Inside unpack8bMode15, mode15_8b = " << mode15_8b << ", theta = " << theta
  //             << ", st1_ring2 = " << st1_ring2  << ", endcap = " << endcap << ", sPhiAB = " << sPhiAB << ", clctA = " << clctA
  //             << ", rpcA = " << rpcA << ", rpcB = " << rpcB << ", rpcC = " << rpcC << ", rpcD = " << rpcD << std::endl;

  if (not(mode15_8b >= 0 && mode15_8b < pow(2, 8)))
    { edm::LogError("L1T") << "mode15_8b = " << mode15_8b; return; }
  if (not(abs(endcap) == 1 && abs(sPhiAB) == 1))
    { edm::LogError("L1T") << "endcap = " << endcap << ", sPhiAB = " << sPhiAB; return; }

  rpcA = 0; rpcB = 0; rpcC = 0; rpcD = 0;

  if (mode15_8b >= 64) st1_ring2 = 1;
  else                 st1_ring2 = 0;

  int rpc_clct, rpc_word, clctA_2b, nRPC = -1;

  if (st1_ring2) {

    rpc_clct = (mode15_8b % 32);
    theta    = (mode15_8b - 64 - rpc_clct) / 32;
    theta   += 8;

    if (rpc_clct < 4) clctA_2b = 0;
    else              clctA_2b = (rpc_clct % 4);
    rpc_word = rpc_clct - clctA_2b;

    // if (clctA_2b != 0) clctA = unpackCLCT(clctA_2b, endcap, sPhiAB, 2);
    clctA = clctA_2b;

    switch (rpc_word) {
    case  0: nRPC = 2; rpcA = 1; rpcB = 1; break;
    case  1: nRPC = 2; rpcA = 1; rpcC = 1; break;
    case  2: nRPC = 2; rpcA = 1; rpcD = 1; break;
    case  3: nRPC = 1; rpcA = 1;           break;
    case  4: nRPC = 2; rpcD = 1; rpcB = 1; break;
    case  8: nRPC = 2; rpcD = 1; rpcC = 1; break;
    case 12: nRPC = 2; rpcB = 1; rpcC = 1; break;
    case 16: nRPC = 1; rpcD = 1;           break;
    case 20: nRPC = 1; rpcB = 1;           break;
    case 24: nRPC = 1; rpcC = 1;           break;
    case 28: nRPC = 0;                     break;
    default: break;
    }
  } // End conditional: if (st1_ring2)
  else {

    rpc_clct  = (mode15_8b % 16);
    theta     = (mode15_8b - rpc_clct) / 16;
    clctA_2b  = (rpc_clct % 4);
    rpc_word  = (rpc_clct - clctA_2b) / 4;

    // if (clctA_2b != 0) clctA = unpackCLCT(clctA_2b, endcap, sPhiAB, 2);
    clctA = clctA_2b;

    switch(rpc_word) {
    case 0: nRPC = 1; theta += 4; rpcD = 1; break;
    case 1: nRPC = 1; theta += 4; rpcC = 1; break;
    case 2: nRPC = 0; theta += 4;           break;
    case 3: nRPC = 0;                       break;
    default: break;
    }
  }

  // std::cout << "  * Output theta = " << theta << ", st1_ring2 = " << st1_ring2 << ", clctA = " << clctA
  //             << ", rpcA = " << rpcA << ", rpcB = " << rpcB << ", rpcC = " << rpcC << ", rpcD = " << rpcD << std::endl;

  if (not(nRPC >= 0))
    { edm::LogError("L1T") << "nRPC = " << nRPC; return; }

} // End function: void PtAssignmentEngineAux2017::unpack8bMode15()
