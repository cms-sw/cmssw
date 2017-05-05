#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.hh"
#include <iostream> // KK: couts below will bring David's attention!
#include <cassert>
#include <cmath>

// // ModeVariables is a 2D arrary indexed by [TrackMode(13 Total Listed Below)][VariableNumber(20 Total Constructed Above)]
// // Variable numbering
// // 0 = dPhi12
// // 1 = dPhi13
// // 2 = dPhi14
// // 3 = dPhi23
// // 4 = dPhi24
// // 5 = dPhi34
// // 6 = dEta12
// // 7 = dEta13
// // 8 = dEta14
// // 9 = dEta23
// // 10 = dEta24
// // 11 = dEta34
// // 12 = CLCT1
// // 13 = CLCT2
// // 14 = CLCT3
// // 15 = CLCT4
// // 16 = CSCID1
// // 17 = CSCID2
// // 18 = CSCID3
// // 19 = CSCID4
// // 20 = FR1
// // 21 = FR2
// // 22 = FR3
// // 23 = FR4

// // Bobby's Scheme3 (or "SchemeC"), with 30 bit compression //
// //3:TrackEta:dPhi12:dEta12:CLCT1:CLCT2:FR1
// //4:Single Station Track Not Possible
// //5:TrackEta:dPhi13:dEta13:CLCT1:CLCT3:FR1
// //6:TrackEta:dPhi23:dEta23:CLCT2:CLCT3:FR2
// //7:TrackEta:dPhi12:dPhi23:dEta13:CLCT1:FR1
// //8:Single Station Track Not Possible
// //9:TrackEta:dPhi14:dEta14:CLCT1:CLCT4:FR1
// //10:TrackEta:dPhi24:dEta24:CLCT2:CLCT4:FR2
// //11:TrackEta:dPhi12:dPhi24:dEta14:CLCT1:FR1
// //12:TrackEta:dPhi34:dEta34:CLCT3:CLCT4:FR3
// //13:TrackEta:dPhi13:dPhi34:dEta14:CLCT1:FR1
// //14:TrackEta:dPhi23:dPhi34:dEta24:CLCT2
// //15:TrackEta:dPhi12:dPhi23:dPhi34:FR1

// static const int ModeVariables_Scheme3[13][6] =
// {
//     {0,6,12,13,20,-999},              // 3
//     {-999,-999,-999,-999,-999,-999},  // 4
//     {1,7,12,14,20,-999},              // 5
//     {3,9,13,14,21,-999},              // 6
//     {0,3,7,12,20,-999},               // 7
//     {-999,-999,-999,-999,-999,-999},  // 8
//     {2,8,12,15,20,-999},              // 9
//     {4,10,13,15,21,-999},             // 10
//     {0,4,8,12,20,-999},               // 11
//     {5,11,14,15,22,-999},             // 12
//     {1,5,8,16,20,-999},               // 13
//     {3,5,10,13,-999,-999},            // 14
//     {0,3,5,20,-999,-999}              // 15
// };


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

// const int (*PtAssignmentEngineAux2017::getModeVariables() const)[6] {
//   return ModeVariables_Scheme3;
// }

int PtAssignmentEngineAux2017::getNLBdPhi(int dPhi, int bits, int max) const {
  assert( (bits == 4 && max == 256) || 
	  (bits == 5 && max == 256) || 
	  (bits == 7 && max == 512) );

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

  assert( abs(sign_) == 1 && dPhi_ >= 0 && dPhi_ < max);
  return (sign_ * dPhi_);
} // End function: nt PtAssignmentEngineAux2017::getNLBdPhi()


int PtAssignmentEngineAux2017::getNLBdPhiBin(int dPhi, int bits, int max) const {
  assert( (bits == 4 && max == 256) || 
	  (bits == 5 && max == 256) || 
	  (bits == 7 && max == 512) );
  
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
  
  assert(dPhiBin_ >= 0 && dPhiBin_ < pow(2, bits));
  return (dPhiBin_);
} // End function: int PtAssignmentEngineAux2017::getNLBdPhiBin()


int PtAssignmentEngineAux2017::getdPhiFromBin(int dPhiBin, int bits, int max) const {
  assert( (bits == 4 && max == 256) || 
	  (bits == 5 && max == 256) || 
	  (bits == 7 && max == 512) );
  
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

  assert(dPhi_ >= 0 && dPhi_ < max);
  return (dPhi_);
} // End function: int PtAssignmentEngineAux2017::getdPhiFromBin()


int PtAssignmentEngineAux2017::getCLCT(int clct, int endcap, int dPhiSign, int bits) const {
  assert( clct >= 0 && clct <= 10 && abs(endcap) == 1 && 
	  abs(dPhiSign) == 1 && (bits == 2 || bits == 3) );

  // Convention here: endcap == +/-1, dPhiSign = +/-1.  May need to change to match FW. - AWB 17.03.17
  int clct_ = 0;
  int sign_ = -1 * endcap * dPhiSign;  // CLCT bend is with dPhi in ME-, opposite in ME+

  if (clct < 2) {
    // std::cout << "\n\n*** In endcap " << endcap << ", CLCT = " << clct << std::endl;
    clct = 2;
  }

  // CLCT pattern can be converted into |bend| x sign as follows:
  // |bend| = (10 + (pattern % 2) - pattern) / 2
  //   * 10 --> 0, 9/8 --> 1, 7/6 --> 2, 5/4 --> 3, 3/2 --> 4, 0 indicates RPC hit
  //  sign  = ((pattern % 2) == 1 ? -1 : 1) * (endcap == 1 ? -1 : 1)   
  //   * In ME+, even CLCTs have negative sign, odd CLCTs have positive

  // For use in all 3- and 4-station modes (7, 11, 13, 14, 15)
  // Bends [-4, -3, -2] --> 0, [-1, 0] --> 1, [+1] --> 2, [+2, +3, +4] --> 3
  if (bits == 2) {
    assert(clct >= 2);
    switch (clct) {
    case 10: clct_ = 1;                  break;
    case  9: clct_ = (sign_ > 0 ? 1 : 2); break;
    case  8: clct_ = (sign_ < 0 ? 1 : 2); break;
    case  7: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  6: clct_ = (sign_ < 0 ? 0 : 3); break;
    case  5: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  4: clct_ = (sign_ < 0 ? 0 : 3); break;
    case  3: clct_ = (sign_ > 0 ? 0 : 3); break;
    case  2: clct_ = (sign_ < 0 ? 0 : 3); break;
    default: clct_ = 0;                   break;
    }
  } // End conditional: if (bits == 2)

  // For use in all 2-station modes (3, 5, 6, 9, 10, 12)
  // Bends [isRPC] --> 0, [-4, -3] --> 1, [-2] --> 2, [-1] --> 3, [0] --> 4, [+1] --> 5, [+2] --> 6, [+3, +4] --> 7
  else if (bits == 3) {
    assert(clct >= 2 || clct == 0);
    switch (clct) {
    case 10: clct_ = 4;                   break;
    case  9: clct_ = (sign_ > 0 ? 3 : 5); break;
    case  8: clct_ = (sign_ < 0 ? 3 : 5); break;
    case  7: clct_ = (sign_ > 0 ? 2 : 6); break;
    case  6: clct_ = (sign_ < 0 ? 2 : 6); break;
    case  5: clct_ = (sign_ > 0 ? 1 : 7); break;
    case  4: clct_ = (sign_ < 0 ? 1 : 7); break;
    case  3: clct_ = (sign_ > 0 ? 1 : 7); break;
    case  2: clct_ = (sign_ < 0 ? 1 : 7); break;
    case  0: clct_ = 0;                   break;
    default: clct_ = 0;                   break;
    }
  } // End conditional: else if (bits == 3)

  assert(clct_ >= 0 && clct_ < pow(2, bits));
  return clct_;
} // End function: int PtAssignmentEngineAux2017::getCLCT()


int PtAssignmentEngineAux2017::getdTheta(int dTheta, int bits) const {
  assert( bits == 2 || bits == 3 );

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
    if      (dTheta ==  0)
      dTheta_ = 4;
    else if (dTheta == +1)
      dTheta_ = 5;
    else if (dTheta == +2)
      dTheta_ = 6;
    else
      dTheta_ = 7;
  } // End conditional: if (bits == 3)

  assert(dTheta_ >= 0 && dTheta_ < pow(2, bits));
  return (dTheta_);
} // End function: int PtAssignmentEngineAux2017::getdTheta()


int PtAssignmentEngineAux2017::getTheta(int theta, int st1_ring2, int bits) const {
  assert( theta >= 5 && theta < 128 && 
	  (st1_ring2 == 0 || st1_ring2 == 1) && 
	  (bits == 4 || bits == 5) );

  int theta_ = -99;

  // For use in mode 15
  if (bits == 4) {
    if (st1_ring2 == 0) {
      // Should never fail with dTheta < 4 windows ... should change to using ME1 for track theta - AWB 17.03.17
      if (theta > 58) {
	std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/1 LCT and track theta = " << theta << std::endl;
      }     
      theta_ = (std::min( std::max(theta, 5), 58) - 5) / 9;
    }
    else if (st1_ring2 == 1) {
      // Should rarely fail with dTheta < 4 windows ... should change to using ME1 for track theta - AWB 17.03.17
      if (theta < 44 || theta > 88) {
	std::cout << "\n\n*** Bizzare case of mode 15 track with ME1/2 LCT and track theta = " << theta << std::endl;
      }
      theta_ = ((std::min( std::max(theta, 44), 88) - 44) / 9) + 6;
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

  assert(theta_ >= 0 && ((bits == 4 && theta_ <= 10) || (bits == 5 && theta_ < pow(2, bits))) );
  return (theta_);
} // End function: int PtAssignmentEngineAux2017::getTheta()


// // Need to re-check / verify this - AWB 17.03.17
// // front-rear LUTs
// // [sector[0]][station 0-4][chamber id]
// // chamber numbers start from 1, so add an extra low bit for invalid chamber = 0
// static const int FRLUT[2][5] = {
//   {0b0000000100100, 0b0000001011010, 0b0101010101010, 0b0010101010100, 0b0010101010100},
//   {0b0000000100100, 0b0000001011010, 0b0111010100100, 0b0000101011010, 0b0000101011010}
// };

// int PtAssignmentEngineAux2017::getFRLUT(int sector, int station, int chamber) const {
//   int bits = FRLUT[(sector-1)%2][station];
//   bool isFront = bits & (1<<chamber);
//   return isFront;
// }
