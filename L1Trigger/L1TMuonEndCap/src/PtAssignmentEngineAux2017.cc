#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"

#include <iostream>

// From here down, exact copy of code used for training BDT: EMTFPtAssign2017/src/PtLUTVarCalc.cc

// Arrays that map the integer dPhi --> dPhi-units. 1/60th of a degree per unit; 255 units --> 4.25 degrees, 511 --> 8.52 degrees

// 256 max units----
// For use in dPhi34 in mode 15.  Derived manually from dPhiNLBMap_5bit_256Max for now; should generate algorithmically. - AWB 17.03.17
static const int dPhiNLBMap_4bit_256Max[16] = {0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 31, 46, 68, 136};

// For use in dPhi23, dPhi24, and dPhi34 in 3- and 4-station modes (7, 11, 13, 14, 15), except for dPhi23 in mode 7 and dPhi34 in mode 15
static const int dPhiNLBMap_5bit_256Max[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                               16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136};
// 512 max units----
// For use in all dPhiAB (where "A" and "B" are the first two stations in the track) in all modes
static const int dPhiNLBMap_7bit_512Max[128] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  71,  72,  73,  74,  75,  76,  77,  79,  80,  81,  83,  84,  86,  87,  89,  91,  92,  94,
    96,  98,  100, 102, 105, 107, 110, 112, 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168,
    174, 181, 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470};

int PtAssignmentEngineAux2017::getNLBdPhi(int dPhi, int bits, int max) const {
  emtf_assert((bits == 4 && max == 256) || (bits == 5 && max == 256) || (bits == 7 && max == 512));

  int dPhi_ = max;
  int sign_ = 1;
  if (dPhi < 0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max == 256) {
    if (bits == 4) {
      dPhi_ = dPhiNLBMap_4bit_256Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_4bit_256Max[edge] <= dPhi && dPhiNLBMap_4bit_256Max[edge + 1] > dPhi) {
          dPhi_ = dPhiNLBMap_4bit_256Max[edge];
          break;
        }
      }
    }  // End conditional: if (bits == 4)
    if (bits == 5) {
      dPhi_ = dPhiNLBMap_5bit_256Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_5bit_256Max[edge] <= dPhi && dPhiNLBMap_5bit_256Max[edge + 1] > dPhi) {
          dPhi_ = dPhiNLBMap_5bit_256Max[edge];
          break;
        }
      }
    }  // End conditional: if (bits == 5)
  }    // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7) {
      dPhi_ = dPhiNLBMap_7bit_512Max[(1 << bits) - 1];
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_7bit_512Max[edge] <= dPhi && dPhiNLBMap_7bit_512Max[edge + 1] > dPhi) {
          dPhi_ = dPhiNLBMap_7bit_512Max[edge];
          break;
        }
      }
    }  // End conditional: if (bits == 7)
  }    // End conditional: else if (max == 512)

  emtf_assert(abs(sign_) == 1 && dPhi_ >= 0 && dPhi_ < max);
  return (sign_ * dPhi_);
}  // End function: int PtAssignmentEngineAux2017::getNLBdPhi()

int PtAssignmentEngineAux2017::getNLBdPhiBin(int dPhi, int bits, int max) const {
  emtf_assert((bits == 4 && max == 256) || (bits == 5 && max == 256) || (bits == 7 && max == 512));

  int dPhiBin_ = (1 << bits) - 1;
  int sign_ = 1;
  if (dPhi < 0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max == 256) {
    if (bits == 4) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_4bit_256Max[edge] <= dPhi && dPhiNLBMap_4bit_256Max[edge + 1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    }  // End conditional: if (bits == 4)
    if (bits == 5) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_5bit_256Max[edge] <= dPhi && dPhiNLBMap_5bit_256Max[edge + 1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    }  // End conditional: if (bits == 5)
  }    // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7) {
      for (int edge = 0; edge < (1 << bits) - 1; edge++) {
        if (dPhiNLBMap_7bit_512Max[edge] <= dPhi && dPhiNLBMap_7bit_512Max[edge + 1] > dPhi) {
          dPhiBin_ = edge;
          break;
        }
      }
    }  // End conditional: if (bits == 7)
  }    // End conditional: else if (max == 512)

  emtf_assert(dPhiBin_ >= 0 && dPhiBin_ < pow(2, bits));
  return (dPhiBin_);
}  // End function: int PtAssignmentEngineAux2017::getNLBdPhiBin()

int PtAssignmentEngineAux2017::getdPhiFromBin(int dPhiBin, int bits, int max) const {
  emtf_assert((bits == 4 && max == 256) || (bits == 5 && max == 256) || (bits == 7 && max == 512));

  int dPhi_ = (1 << bits) - 1;

  if (dPhiBin > (1 << bits) - 1)
    dPhiBin = (1 << bits) - 1;

  if (max == 256) {
    if (bits == 4)
      dPhi_ = dPhiNLBMap_4bit_256Max[dPhiBin];
    if (bits == 5)
      dPhi_ = dPhiNLBMap_5bit_256Max[dPhiBin];
  }  // End conditional: if (max == 256)

  else if (max == 512) {
    if (bits == 7)
      dPhi_ = dPhiNLBMap_7bit_512Max[dPhiBin];
  }  // End conditional: else if (max == 512)

  emtf_assert(dPhi_ >= 0 && dPhi_ < max);
  return (dPhi_);
}  // End function: int PtAssignmentEngineAux2017::getdPhiFromBin()

int PtAssignmentEngineAux2017::getCLCT(int clct, int endcap, int dPhiSign, int bits) const {
  // std::cout << "Inside getCLCT: clct = " << clct << ", endcap = " << endcap
  //             << ", dPhiSign = " << dPhiSign << ", bits = " << bits << std::endl;

  emtf_assert(clct >= 0 && clct <= 10 && abs(endcap) == 1 && abs(dPhiSign) == 1 && (bits == 2 || bits == 3));

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
      case 10:
        clct_ = 1;
        break;
      case 9:
        clct_ = (sign_ > 0 ? 1 : 2);
        break;
      case 8:
        clct_ = (sign_ > 0 ? 2 : 1);
        break;
      case 7:
        clct_ = (sign_ > 0 ? 0 : 3);
        break;
      case 6:
        clct_ = (sign_ > 0 ? 3 : 0);
        break;
      case 5:
        clct_ = (sign_ > 0 ? 0 : 3);
        break;
      case 4:
        clct_ = (sign_ > 0 ? 3 : 0);
        break;
      case 3:
        clct_ = (sign_ > 0 ? 0 : 3);
        break;
      case 2:
        clct_ = (sign_ > 0 ? 3 : 0);
        break;
      case 1:
        clct_ = (sign_ > 0 ? 0 : 3);
        break;
      case 0:
        clct_ = 0;
        break;
      default:
        clct_ = 1;
        break;
    }
  }  // End conditional: if (bits == 2)

  // For use in all 2-station modes (3, 5, 6, 9, 10, 12)
  // Bends [isRPC] --> 0, [-4, -3] --> 1, [-2] --> 2, [-1] --> 3, [0] --> 4, [+1] --> 5, [+2] --> 6, [+3, +4] --> 7
  else if (bits == 3) {
    switch (clct) {
      case 10:
        clct_ = 4;
        break;
      case 9:
        clct_ = (sign_ > 0 ? 3 : 5);
        break;
      case 8:
        clct_ = (sign_ > 0 ? 5 : 3);
        break;
      case 7:
        clct_ = (sign_ > 0 ? 2 : 6);
        break;
      case 6:
        clct_ = (sign_ > 0 ? 6 : 2);
        break;
      case 5:
        clct_ = (sign_ > 0 ? 1 : 7);
        break;
      case 4:
        clct_ = (sign_ > 0 ? 7 : 1);
        break;
      case 3:
        clct_ = (sign_ > 0 ? 1 : 7);
        break;
      case 2:
        clct_ = (sign_ > 0 ? 7 : 1);
        break;
      case 1:
        clct_ = (sign_ > 0 ? 1 : 7);
        break;
      case 0:
        clct_ = 0;
        break;
      default:
        clct_ = 4;
        break;
    }
  }  // End conditional: else if (bits == 3)

  // std::cout << "  * Output clct_ = " << clct_ << std::endl;

  emtf_assert(clct_ >= 0 && clct_ < pow(2, bits));
  return clct_;
}  // End function: int PtAssignmentEngineAux2017::getCLCT()

int PtAssignmentEngineAux2017::unpackCLCT(int clct, int endcap, int dPhiSign, int bits) const {
  // std::cout << "Inside unpackCLCT: clct = " << clct << ", endcap = " << endcap
  //             << ", dPhiSign = " << dPhiSign << ", bits = " << bits << std::endl;

  emtf_assert(bits == 2 || bits == 3);
  emtf_assert(clct >= 0 && clct < pow(2, bits));
  emtf_assert(abs(dPhiSign) == 1);

  // Convention here: endcap == +/-1, dPhiSign = +/-1.
  int clct_ = -1;
  int sign_ = -1 * endcap * dPhiSign;  // CLCT bend is with dPhi in ME-, opposite in ME+

  if (bits == 2) {
    switch (clct) {
      case 1:
        clct_ = 10;
        break;
      case 2:
        clct_ = (sign_ > 0 ? 8 : 9);
        break;
      case 3:
        clct_ = (sign_ > 0 ? 4 : 5);
        break;
      case 0:
        clct_ = 0;
        break;
      default:
        break;
    }
  } else if (bits == 3) {
    switch (clct) {
      case 4:
        clct_ = 10;
        break;
      case 5:
        clct_ = (sign_ > 0 ? 8 : 9);
        break;
      case 3:
        clct_ = (sign_ > 0 ? 9 : 8);
        break;
      case 6:
        clct_ = (sign_ > 0 ? 6 : 7);
        break;
      case 2:
        clct_ = (sign_ > 0 ? 7 : 6);
        break;
      case 7:
        clct_ = (sign_ > 0 ? 4 : 5);
        break;
      case 1:
        clct_ = (sign_ > 0 ? 5 : 4);
        break;
      case 0:
        clct_ = 0;
        break;
      default:
        break;
    }
  }

  // std::cout << "  * Output clct_ = " << clct_ << std::endl;

  emtf_assert(clct_ >= 0 && clct_ <= 10);
  return clct_;
}  // End function: int PtAssignmentEngineAux2017::unpackCLCT()

int PtAssignmentEngineAux2017::getdTheta(int dTheta, int bits) const {
  emtf_assert(bits == 2 || bits == 3);

  int dTheta_ = -99;

  // For use in mode 15
  if (bits == 2) {
    if (abs(dTheta) <= 1)
      dTheta_ = 2;
    else if (abs(dTheta) <= 2)
      dTheta_ = 1;
    else if (dTheta <= -3)
      dTheta_ = 0;
    else
      dTheta_ = 3;
  }  // End conditional: if (bits == 2)

  // For use in all 2- and 3-station modes (all modes except 15)
  else if (bits == 3) {
    if (dTheta <= -4)
      dTheta_ = 0;
    else if (dTheta == -3)
      dTheta_ = 1;
    else if (dTheta == -2)
      dTheta_ = 2;
    else if (dTheta == -1)
      dTheta_ = 3;
    else if (dTheta == 0)
      dTheta_ = 4;
    else if (dTheta == +1)
      dTheta_ = 5;
    else if (dTheta == +2)
      dTheta_ = 6;
    else
      dTheta_ = 7;
  }  // End conditional: if (bits == 3)

  emtf_assert(dTheta_ >= 0 && dTheta_ < pow(2, bits));
  return (dTheta_);
}  // End function: int PtAssignmentEngineAux2017::getdTheta()

int PtAssignmentEngineAux2017::unpackdTheta(int dTheta, int bits) const {
  emtf_assert(bits == 2 || bits == 3);

  int dTheta_ = -99;

  if (bits == 2) {  // For use in mode 15
    switch (dTheta) {
      case 2:
        dTheta_ = 0;
        break;
      case 1:
        dTheta_ = -2;
        break;
      case 0:
        dTheta_ = -3;
        break;
      case 3:
        dTheta_ = 3;
        break;
      default:
        break;
    }
  } else if (bits == 3) {  // For use in all 2- and 3-station modes (all modes except 15)
    switch (dTheta) {
      case 0:
        dTheta_ = -4;
        break;
      case 1:
        dTheta_ = -3;
        break;
      case 2:
        dTheta_ = -2;
        break;
      case 3:
        dTheta_ = -1;
        break;
      case 4:
        dTheta_ = 0;
        break;
      case 5:
        dTheta_ = 1;
        break;
      case 6:
        dTheta_ = 2;
        break;
      case 7:
        dTheta_ = 3;
        break;
      default:
        break;
    }
  }

  emtf_assert(dTheta_ >= -4 && dTheta_ <= 3);
  return (dTheta_);
}  // End function: int PtAssignmentEngineAux2017::unpackdTheta(int dTheta, int bits)

int PtAssignmentEngineAux2017::getTheta(int theta, int st1_ring2, int bits) const {
  emtf_assert(theta >= 5 && theta < 128 && (st1_ring2 == 0 || st1_ring2 == 1) && (bits == 4 || bits == 5));

  int theta_ = -99;

  // For use in mode 15
  if (bits == 4) {
    if (st1_ring2 == 0) {
      // Should rarely fail ... should change to using ME1 for track theta - AWB 05.06.17
      if (theta > 52) {
        // std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/1 LCT and track theta = " << theta << std::endl;
      }
      theta_ = (std::min(std::max(theta, 5), 52) - 5) / 6;
    } else if (st1_ring2 == 1) {
      // Should rarely fail ... should change to using ME1 for track theta - AWB 05.06.17
      if (theta < 46 || theta > 87) {
        // std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/2 LCT and track theta = " << theta << std::endl;
      }
      theta_ = ((std::min(std::max(theta, 46), 87) - 46) / 7) + 8;
    }
  }  // End conditional: if (bits == 4)

  // For use in all 2- and 3-station modes (all modes except 15)
  else if (bits == 5) {
    if (st1_ring2 == 0) {
      theta_ = (std::max(theta, 1) - 1) / 4;
    } else if (st1_ring2 == 1) {
      theta_ = ((std::min(theta, 104) - 1) / 4) + 6;
    }
  }  // End conditional: else if (bits == 5)

  emtf_assert(theta_ >= 0 && ((bits == 4 && theta_ <= 13) || (bits == 5 && theta_ < pow(2, bits))));
  return (theta_);
}  // End function: int PtAssignmentEngineAux2017::getTheta()

void PtAssignmentEngineAux2017::unpackTheta(int& theta, int& st1_ring2, int bits) const {
  emtf_assert(bits == 4 || bits == 5);
  emtf_assert(theta >= 0 && theta < pow(2, bits));

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

  emtf_assert(theta >= 5 && theta <= 104);

}  // End function: void PtAssignmentEngineAux2017::unpackTheta()

int PtAssignmentEngineAux2017::unpackSt1Ring2(int theta, int bits) const {
  emtf_assert(bits == 4 || bits == 5);
  emtf_assert(theta >= 0 && theta < pow(2, bits));

  // For use in mode 15
  if (bits == 4) {
    if (theta < 6)
      return 0;
    else
      return 1;
  } else {
    if (theta < 15)
      return 0;
    else
      return 1;
  }

}  // End function: void PtAssignmentEngineAux2017::unpackSt1Ring2()

int PtAssignmentEngineAux2017::get2bRPC(int clctA, int clctB, int clctC) const {
  int rpc_2b = -99;

  if (clctA == 0)
    rpc_2b = 0;
  else if (clctC == 0)
    rpc_2b = 1;
  else if (clctB == 0)
    rpc_2b = 2;
  else
    rpc_2b = 3;

  emtf_assert(rpc_2b >= 0 && rpc_2b < 4);
  return (rpc_2b);
}  // End function: int PtAssignmentEngineAux2017::get2bRPC()

void PtAssignmentEngineAux2017::unpack2bRPC(int rpc_2b, int& rpcA, int& rpcB, int& rpcC) const {
  emtf_assert(rpc_2b >= 0 && rpc_2b < 4);

  rpcA = 0;
  rpcB = 0;
  rpcC = 0;

  if (rpc_2b == 0)
    rpcA = 1;
  else if (rpc_2b == 1)
    rpcC = 1;
  else if (rpc_2b == 2)
    rpcB = 1;

}  // End function: int PtAssignmentEngineAux2017::unpack2bRPC()

int PtAssignmentEngineAux2017::get8bMode15(
    int theta, int st1_ring2, int endcap, int sPhiAB, int clctA, int clctB, int clctC, int clctD) const {
  // std::cout << "Inside get8bMode15, theta = " << theta << ", st1_ring2 = " << st1_ring2 << ", endcap = " << endcap << ", sPhiAB = " << sPhiAB
  //             << ", clctA = " << clctA << ", clctB = " << clctB << ", clctC = " << clctC << ", clctD = " << clctD << std::endl;

  if (st1_ring2)
    theta = (std::min(std::max(theta, 46), 87) - 46) / 7;
  else
    theta = (std::min(std::max(theta, 5), 52) - 5) / 6;

  emtf_assert(theta >= 0 && theta < 10);

  int clctA_2b = getCLCT(clctA, endcap, sPhiAB, 2);

  int nRPC = (clctA == 0) + (clctB == 0) + (clctC == 0) + (clctD == 0);
  int rpc_word, rpc_clct, mode15_8b;

  if (st1_ring2) {
    if (nRPC >= 2 && clctA == 0 && clctB == 0)
      rpc_word = 0;
    else if (nRPC >= 2 && clctA == 0 && clctC == 0)
      rpc_word = 1;
    else if (nRPC >= 2 && clctA == 0 && clctD == 0)
      rpc_word = 2;
    else if (nRPC == 1 && clctA == 0)
      rpc_word = 3;
    else if (nRPC >= 2 && clctD == 0 && clctB == 0)
      rpc_word = 4;
    else if (nRPC >= 2 && clctD == 0 && clctC == 0)
      rpc_word = 8;
    else if (nRPC >= 2 && clctB == 0 && clctC == 0)
      rpc_word = 12;
    else if (nRPC == 1 && clctD == 0)
      rpc_word = 16;
    else if (nRPC == 1 && clctB == 0)
      rpc_word = 20;
    else if (nRPC == 1 && clctC == 0)
      rpc_word = 24;
    else
      rpc_word = 28;
    rpc_clct = rpc_word + clctA_2b;
    mode15_8b = (theta * 32) + rpc_clct + 64;
  } else {
    if (theta >= 4 && clctD == 0)
      rpc_word = 0;
    else if (theta >= 4 && clctC == 0)
      rpc_word = 1;
    else if (theta >= 4)
      rpc_word = 2;
    else
      rpc_word = 3;
    rpc_clct = rpc_word * 4 + clctA_2b;
    mode15_8b = ((theta % 4) * 16) + rpc_clct;
  }

  // std::cout << "  * Output mode15_8b = " << mode15_8b << std::endl;

  emtf_assert(mode15_8b >= 0 && mode15_8b < pow(2, 8));
  return (mode15_8b);

}  // End function: int PtAssignmentEngineAux2017::get8bMode15()

void PtAssignmentEngineAux2017::unpack8bMode15(int mode15_8b,
                                               int& theta,
                                               int& st1_ring2,
                                               int endcap,
                                               int sPhiAB,
                                               int& clctA,
                                               int& rpcA,
                                               int& rpcB,
                                               int& rpcC,
                                               int& rpcD) const {
  // std::cout << "Inside unpack8bMode15, mode15_8b = " << mode15_8b << ", theta = " << theta
  //             << ", st1_ring2 = " << st1_ring2  << ", endcap = " << endcap << ", sPhiAB = " << sPhiAB << ", clctA = " << clctA
  //             << ", rpcA = " << rpcA << ", rpcB = " << rpcB << ", rpcC = " << rpcC << ", rpcD = " << rpcD << std::endl;

  emtf_assert(mode15_8b >= 0 && mode15_8b < pow(2, 8));
  emtf_assert(abs(endcap) == 1 && abs(sPhiAB) == 1);

  rpcA = 0;
  rpcB = 0;
  rpcC = 0;
  rpcD = 0;

  if (mode15_8b >= 64)
    st1_ring2 = 1;
  else
    st1_ring2 = 0;

  int rpc_clct, rpc_word, clctA_2b, nRPC = -1;

  if (st1_ring2) {
    rpc_clct = (mode15_8b % 32);
    theta = (mode15_8b - 64 - rpc_clct) / 32;
    theta += 8;

    if (rpc_clct < 4)
      clctA_2b = 0;
    else
      clctA_2b = (rpc_clct % 4);
    rpc_word = rpc_clct - clctA_2b;

    // if (clctA_2b != 0) clctA = unpackCLCT(clctA_2b, endcap, sPhiAB, 2);
    clctA = clctA_2b;

    switch (rpc_word) {
      case 0:
        nRPC = 2;
        rpcA = 1;
        rpcB = 1;
        break;
      case 1:
        nRPC = 2;
        rpcA = 1;
        rpcC = 1;
        break;
      case 2:
        nRPC = 2;
        rpcA = 1;
        rpcD = 1;
        break;
      case 3:
        nRPC = 1;
        rpcA = 1;
        break;
      case 4:
        nRPC = 2;
        rpcD = 1;
        rpcB = 1;
        break;
      case 8:
        nRPC = 2;
        rpcD = 1;
        rpcC = 1;
        break;
      case 12:
        nRPC = 2;
        rpcB = 1;
        rpcC = 1;
        break;
      case 16:
        nRPC = 1;
        rpcD = 1;
        break;
      case 20:
        nRPC = 1;
        rpcB = 1;
        break;
      case 24:
        nRPC = 1;
        rpcC = 1;
        break;
      case 28:
        nRPC = 0;
        break;
      default:
        break;
    }
  }  // End conditional: if (st1_ring2)
  else {
    rpc_clct = (mode15_8b % 16);
    theta = (mode15_8b - rpc_clct) / 16;
    clctA_2b = (rpc_clct % 4);
    rpc_word = (rpc_clct - clctA_2b) / 4;

    // if (clctA_2b != 0) clctA = unpackCLCT(clctA_2b, endcap, sPhiAB, 2);
    clctA = clctA_2b;

    switch (rpc_word) {
      case 0:
        nRPC = 1;
        theta += 4;
        rpcD = 1;
        break;
      case 1:
        nRPC = 1;
        theta += 4;
        rpcC = 1;
        break;
      case 2:
        nRPC = 0;
        theta += 4;
        break;
      case 3:
        nRPC = 0;
        break;
      default:
        break;
    }
  }

  // std::cout << "  * Output theta = " << theta << ", st1_ring2 = " << st1_ring2 << ", clctA = " << clctA
  //             << ", rpcA = " << rpcA << ", rpcB = " << rpcB << ", rpcC = " << rpcC << ", rpcD = " << rpcD << std::endl;

  emtf_assert(nRPC >= 0);

}  // End function: void PtAssignmentEngineAux2017::unpack8bMode15()

// _____________________________________________________________________________
// From here down, code was originally in PtLUTVarCalc.cc

int PtAssignmentEngineAux2017::calcTrackTheta(const int th1,
                                              const int th2,
                                              const int th3,
                                              const int th4,
                                              const int st1_ring2,
                                              const int mode,
                                              const bool BIT_COMP) const {
  int theta = -99;

  if ((mode % 8) / 4 > 0)  // Has station 2 hit
    theta = th2;
  else if ((mode % 4) / 2 > 0)  // Has station 3 hit
    theta = th3;
  else if ((mode % 2) > 0)  // Has station 4 hit
    theta = th4;

  emtf_assert(theta > 0);

  if (BIT_COMP) {
    int nBits = (mode == 15 ? 4 : 5);
    theta = getTheta(theta, st1_ring2, nBits);
  }

  return theta;
}

void PtAssignmentEngineAux2017::calcDeltaPhis(int& dPh12,
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
                                              const bool BIT_COMP) const {
  dPh12 = ph2 - ph1;
  dPh13 = ph3 - ph1;
  dPh14 = ph4 - ph1;
  dPh23 = ph3 - ph2;
  dPh24 = ph4 - ph2;
  dPh34 = ph4 - ph3;
  dPhSign = 0;

  if (mode >= 8) {           // First hit is station 1
    if ((mode % 8) / 4 > 0)  // Has station 2 hit
      dPhSign = (dPh12 >= 0 ? +1 : -1);
    else if ((mode % 4) / 2 > 0)  // Has station 3 hit
      dPhSign = (dPh13 >= 0 ? +1 : -1);
    else if ((mode % 2) > 0)  // Has station 4 hit
      dPhSign = (dPh14 >= 0 ? +1 : -1);
  } else if ((mode % 8) / 4 > 0) {  // First hit is station 2
    if ((mode % 4) / 2 > 0)         // Has station 3 hit
      dPhSign = (dPh23 >= 0 ? +1 : -1);
    else if ((mode % 2) > 0)  // Has station 4 hit
      dPhSign = (dPh24 >= 0 ? +1 : -1);
  } else if ((mode % 4) / 2 > 0) {  // First hit is station 3
    if ((mode % 2) > 0)             // Has station 4 hit
      dPhSign = (dPh34 >= 0 ? +1 : -1);
  }

  emtf_assert(dPhSign != 0);

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

    dPh12 = getNLBdPhi(dPh12, nBitsA, maxA);
    dPh13 = getNLBdPhi(dPh13, nBitsA, maxA);
    dPh14 = getNLBdPhi(dPh14, nBitsA, maxA);
    if (mode == 7)
      dPh23 = getNLBdPhi(dPh23, nBitsA, maxA);
    else
      dPh23 = getNLBdPhi(dPh23, nBitsB, maxB);
    dPh24 = getNLBdPhi(dPh24, nBitsB, maxB);
    dPh34 = getNLBdPhi(dPh34, nBitsC, maxC);

    // Some delta phi values must be computed from others
    switch (mode) {
      case 15:
        dPh13 = dPh12 + dPh23;
        dPh14 = dPh13 + dPh34;
        dPh24 = dPh23 + dPh34;
        break;
      case 14:
        dPh13 = dPh12 + dPh23;
        break;
      case 13:
        dPh14 = dPh12 + dPh24;
        break;
      case 11:
        dPh14 = dPh13 + dPh34;
        break;
      case 7:
        dPh24 = dPh23 + dPh34;
        break;
      default:
        break;
    }

  }  // End conditional: if (BIT_COMP)

  // Compute summed quantities
  if (mode == 15)
    calcDeltaPhiSums(dPhSum4, dPhSum4A, dPhSum3, dPhSum3A, outStPh, dPh12, dPh13, dPh14, dPh23, dPh24, dPh34);

}  // End function: void PtAssignmentEngineAux2017::calcDeltaPhis()

void PtAssignmentEngineAux2017::calcDeltaThetas(int& dTh12,
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
                                                const bool BIT_COMP) const {
  dTh12 = th2 - th1;
  dTh13 = th3 - th1;
  dTh14 = th4 - th1;
  dTh23 = th3 - th2;
  dTh24 = th4 - th2;
  dTh34 = th4 - th3;

  if (BIT_COMP) {
    int nBits = (mode == 15 ? 2 : 3);

    dTh12 = getdTheta(dTh12, nBits);
    dTh13 = getdTheta(dTh13, nBits);
    dTh14 = getdTheta(dTh14, nBits);
    dTh23 = getdTheta(dTh23, nBits);
    dTh24 = getdTheta(dTh24, nBits);
    dTh34 = getdTheta(dTh34, nBits);
  }  // End conditional: if (BIT_COMP)

}  // Enf function: void PtAssignmentEngineAux2017::calcDeltaThetas()

void PtAssignmentEngineAux2017::calcBends(int& bend1,
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
                                          const bool BIT_COMP) const {
  bend1 = calcBendFromPattern(pat1, endcap);
  bend2 = calcBendFromPattern(pat2, endcap);
  bend3 = calcBendFromPattern(pat3, endcap);
  bend4 = calcBendFromPattern(pat4, endcap);

  if (BIT_COMP) {
    int nBits = 3;
    if (mode == 7 || mode == 11 || mode > 12)
      nBits = 2;

    if (mode / 8 > 0)  // Has station 1 hit
      bend1 = getCLCT(pat1, endcap, dPhSign, nBits);
    if ((mode % 8) / 4 > 0)  // Has station 2 hit
      bend2 = getCLCT(pat2, endcap, dPhSign, nBits);
    if ((mode % 4) / 2 > 0)  // Has station 3 hit
      bend3 = getCLCT(pat3, endcap, dPhSign, nBits);
    if ((mode % 2) > 0)  // Has station 4 hit
      bend4 = getCLCT(pat4, endcap, dPhSign, nBits);
  }  // End conditional: if (BIT_COMP)

}  // End function: void PtAssignmentEngineAux2017::calcBends()

void PtAssignmentEngineAux2017::calcRPCs(int& RPC1,
                                         int& RPC2,
                                         int& RPC3,
                                         int& RPC4,
                                         const int mode,
                                         const int st1_ring2,
                                         const int theta,
                                         const bool BIT_COMP) const {
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
      if (mode == 15) {
        if (RPC1 == 1 && RPC2 == 1) {
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
        if (RPC1 == 1) {
          RPC2 = 0;
          RPC3 = 0;
        } else if (RPC3 == 1) {
          RPC2 = 0;
        }
      } else if (mode == 13) {
        if (RPC1 == 1) {
          RPC2 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC2 = 0;
        }
      } else if (mode == 11) {
        if (RPC1 == 1) {
          RPC3 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC3 = 0;
        }
      } else if (mode == 7) {
        if (RPC2 == 1) {
          RPC3 = 0;
          RPC4 = 0;
        } else if (RPC4 == 1) {
          RPC3 = 0;
        }
      }

    }  // End conditional: if (nRPC >= 2)
  }    // End conditional: if (BIT_COMP)

}  // End function: void PtAssignmentEngineAux2017::calcRPCs()

int PtAssignmentEngineAux2017::calcBendFromPattern(const int pattern, const int endcap) const {
  int bend = -99;
  if (pattern < 0)
    return bend;

  if (pattern == 10)
    bend = 0;
  else if ((pattern % 2) == 0)
    bend = (10 - pattern) / 2;
  else if ((pattern % 2) == 1)
    bend = -1 * (11 - pattern) / 2;

  // Reverse to match dPhi convention
  if (endcap == 1)
    bend *= -1;

  emtf_assert(bend != -99);
  return bend;
}

void PtAssignmentEngineAux2017::calcDeltaPhiSums(int& dPhSum4,
                                                 int& dPhSum4A,
                                                 int& dPhSum3,
                                                 int& dPhSum3A,
                                                 int& outStPh,
                                                 const int dPh12,
                                                 const int dPh13,
                                                 const int dPh14,
                                                 const int dPh23,
                                                 const int dPh24,
                                                 const int dPh34) const {
  dPhSum4 = dPh12 + dPh13 + dPh14 + dPh23 + dPh24 + dPh34;
  dPhSum4A = abs(dPh12) + abs(dPh13) + abs(dPh14) + abs(dPh23) + abs(dPh24) + abs(dPh34);
  int devSt1 = abs(dPh12) + abs(dPh13) + abs(dPh14);
  int devSt2 = abs(dPh12) + abs(dPh23) + abs(dPh24);
  int devSt3 = abs(dPh13) + abs(dPh23) + abs(dPh34);
  int devSt4 = abs(dPh14) + abs(dPh24) + abs(dPh34);

  if (devSt4 > devSt3 && devSt4 > devSt2 && devSt4 > devSt1)
    outStPh = 4;
  else if (devSt3 > devSt4 && devSt3 > devSt2 && devSt3 > devSt1)
    outStPh = 3;
  else if (devSt2 > devSt4 && devSt2 > devSt3 && devSt2 > devSt1)
    outStPh = 2;
  else if (devSt1 > devSt4 && devSt1 > devSt3 && devSt1 > devSt2)
    outStPh = 1;
  else
    outStPh = 0;

  if (outStPh == 4) {
    dPhSum3 = dPh12 + dPh13 + dPh23;
    dPhSum3A = abs(dPh12) + abs(dPh13) + abs(dPh23);
  } else if (outStPh == 3) {
    dPhSum3 = dPh12 + dPh14 + dPh24;
    dPhSum3A = abs(dPh12) + abs(dPh14) + abs(dPh24);
  } else if (outStPh == 2) {
    dPhSum3 = dPh13 + dPh14 + dPh34;
    dPhSum3A = abs(dPh13) + abs(dPh14) + abs(dPh34);
  } else {
    dPhSum3 = dPh23 + dPh24 + dPh34;
    dPhSum3A = abs(dPh23) + abs(dPh24) + abs(dPh34);
  }

}  // End function: void PtAssignmentEngineAux2017::calcDeltaPhiSums()
