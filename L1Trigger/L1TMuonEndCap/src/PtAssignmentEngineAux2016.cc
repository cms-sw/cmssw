#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2016.h"

//ModeVariables is a 2D arrary indexed by [TrackMode(13 Total Listed Below)][VariableNumber(20 Total Constructed Above)]
// Variable numbering
// 0 = dPhi12
// 1 = dPhi13
// 2 = dPhi14
// 3 = dPhi23
// 4 = dPhi24
// 5 = dPhi34
// 6 = dEta12
// 7 = dEta13
// 8 = dEta14
// 9 = dEta23
// 10 = dEta24
// 11 = dEta34
// 12 = CLCT1
// 13 = CLCT2
// 14 = CLCT3
// 15 = CLCT4
// 16 = CSCID1
// 17 = CSCID2
// 18 = CSCID3
// 19 = CSCID4
// 20 = FR1
// 21 = FR2
// 22 = FR3
// 23 = FR4

// Bobby's Scheme3 (or "SchemeC"), with 30 bit compression //
//3:TrackEta:dPhi12:dEta12:CLCT1:CLCT2:FR1
//4:Single Station Track Not Possible
//5:TrackEta:dPhi13:dEta13:CLCT1:CLCT3:FR1
//6:TrackEta:dPhi23:dEta23:CLCT2:CLCT3:FR2
//7:TrackEta:dPhi12:dPhi23:dEta13:CLCT1:FR1
//8:Single Station Track Not Possible
//9:TrackEta:dPhi14:dEta14:CLCT1:CLCT4:FR1
//10:TrackEta:dPhi24:dEta24:CLCT2:CLCT4:FR2
//11:TrackEta:dPhi12:dPhi24:dEta14:CLCT1:FR1
//12:TrackEta:dPhi34:dEta34:CLCT3:CLCT4:FR3
//13:TrackEta:dPhi13:dPhi34:dEta14:CLCT1:FR1
//14:TrackEta:dPhi23:dPhi34:dEta24:CLCT2
//15:TrackEta:dPhi12:dPhi23:dPhi34:FR1

static const int ModeVariables_Scheme3[13][6] =
{
    {0,6,12,13,20,-999},              // 3
    {-999,-999,-999,-999,-999,-999},  // 4
    {1,7,12,14,20,-999},              // 5
    {3,9,13,14,21,-999},              // 6
    {0,3,7,12,20,-999},               // 7
    {-999,-999,-999,-999,-999,-999},  // 8
    {2,8,12,15,20,-999},              // 9
    {4,10,13,15,21,-999},             // 10
    {0,4,8,12,20,-999},               // 11
    {5,11,14,15,22,-999},             // 12
    {1,5,8,16,20,-999},               // 13
    {3,5,10,13,-999,-999},            // 14
    {0,3,5,20,-999,-999}              // 15
};

// 256 max units----

static const int dPhiNLBMap_5bit_256Max[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 28, 31, 34, 39, 46, 55, 68, 91, 136};

static const int dPhiNLBMap_6bit_256Max[64] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 45, 47, 49, 51, 53, 56, 58, 61, 65, 68, 73, 78, 83, 89, 97, 106, 116, 129, 145, 166, 193, 232};

static const int dPhiNLBMap_7bit_256Max[128] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 90, 91, 93, 94, 96, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 118, 120, 123, 125, 128, 131, 134, 138, 141, 145, 149, 153, 157, 161, 166, 171, 176, 182, 188, 194, 201, 209, 217, 225, 235, 245};

// 512 max units----

static const int dPhiNLBMap_7bit_512Max[128] =  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 86, 87, 89, 91, 92, 94, 96, 98, 100, 102, 105, 107, 110, 112, 115, 118, 121, 124, 127, 131, 135, 138, 143, 147, 152, 157, 162, 168, 174, 181, 188, 196, 204, 214, 224, 235, 247, 261, 276, 294, 313, 336, 361, 391, 427, 470};

static const int dPhiNLBMap_8bit_512Max[256] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170, 171, 172, 174, 175, 176, 178, 179, 180, 182, 183, 185, 186, 188, 190, 191, 193, 194, 196, 198, 200, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 228, 230, 232, 235, 237, 240, 242, 245, 248, 250, 253, 256, 259, 262, 265, 268, 272, 275, 278, 282, 285, 289, 293, 297, 300, 305, 309, 313, 317, 322, 327, 331, 336, 341, 347, 352, 358, 363, 369, 375, 382, 388, 395, 402, 410, 417, 425, 433, 442, 450, 460, 469, 479, 490, 500};


const int (*PtAssignmentEngineAux2016::getModeVariables() const)[6] {
  return ModeVariables_Scheme3;
}

int PtAssignmentEngineAux2016::getNLBdPhi(int dPhi, int bits, int max) const {
  int dPhi_= max;
  int sign_ = 1;
  if (dPhi<0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max==256)
  {
    if (bits == 5)
    {
      dPhi_ = dPhiNLBMap_5bit_256Max[(1<<bits)-1];
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_5bit_256Max[edge]<=dPhi && dPhiNLBMap_5bit_256Max[edge+1]>dPhi)
        {
          dPhi_ = dPhiNLBMap_5bit_256Max[edge];
          break;
        }
      }
    }
    if (bits == 6)
    {
      dPhi_ = dPhiNLBMap_6bit_256Max[(1<<bits)-1];
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_6bit_256Max[edge]<=dPhi && dPhiNLBMap_6bit_256Max[edge+1]>dPhi)
        {
          dPhi_ = dPhiNLBMap_6bit_256Max[edge];
          break;
        }
      }
    }
    if (bits == 7)
    {
      dPhi_ = dPhiNLBMap_7bit_256Max[(1<<bits)-1];
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_7bit_256Max[edge]<=dPhi && dPhiNLBMap_7bit_256Max[edge+1]>dPhi)
        {
          dPhi_ = dPhiNLBMap_7bit_256Max[edge];
          break;
        }
      }
    }
  }

  else if (max==512)
  {
    if (bits == 7)
    {
      dPhi_ = dPhiNLBMap_7bit_512Max[(1<<bits)-1];
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_7bit_512Max[edge]<=dPhi && dPhiNLBMap_7bit_512Max[edge+1]>dPhi)
        {
          dPhi_ = dPhiNLBMap_7bit_512Max[edge];
          break;
        }
      }
    }
    if (bits == 8)
    {
      dPhi_ = dPhiNLBMap_8bit_512Max[(1<<bits)-1];
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_8bit_512Max[edge]<=dPhi && dPhiNLBMap_8bit_512Max[edge+1]>dPhi)
        {
          dPhi_ = dPhiNLBMap_8bit_512Max[edge];
          break;
        }
      }
    }
  }

  if (dPhi>=max) dPhi_ = max;
  return (sign_ * dPhi_);
}

int PtAssignmentEngineAux2016::getNLBdPhiBin(int dPhi, int bits, int max) const {
  int dPhiBin_= (1<<bits)-1;
  int sign_ = 1;
  if (dPhi<0)
    sign_ = -1;
  dPhi = sign_ * dPhi;

  if (max==256)
  {
    if (bits == 5)
    {
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_5bit_256Max[edge]<=dPhi && dPhiNLBMap_5bit_256Max[edge+1]>dPhi)
        {
          dPhiBin_ = edge;
          break;
        }
      }
    }
    if (bits == 6)
    {
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_6bit_256Max[edge]<=dPhi && dPhiNLBMap_6bit_256Max[edge+1]>dPhi)
        {
          dPhiBin_ = edge;
          break;
        }
      }
    }
    if (bits == 7)
    {
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_7bit_256Max[edge]<=dPhi && dPhiNLBMap_7bit_256Max[edge+1]>dPhi)
        {
          dPhiBin_ = edge;
          break;
        }
      }
    }
  }

  else if (max==512)
  {
    if (bits == 7)
    {
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_7bit_512Max[edge]<=dPhi && dPhiNLBMap_7bit_512Max[edge+1]>dPhi)
        {
          dPhiBin_ = edge;
          break;
        }
      }
    }
    if (bits == 8)
    {
      for (int edge=0; edge<(1<<bits)-1; edge++)
      {
        if (dPhiNLBMap_8bit_512Max[edge]<=dPhi && dPhiNLBMap_8bit_512Max[edge+1]>dPhi)
        {
          dPhiBin_ = edge;
          break;
        }
      }
    }
  }

  return (dPhiBin_);
}

int PtAssignmentEngineAux2016::getdPhiFromBin(int dPhiBin, int bits, int max) const {
  int dPhi_= (1<<bits)-1;

  if (dPhiBin>(1<<bits)-1)
    dPhiBin = (1<<bits)-1;

  if (max==256)
  {
    if (bits == 5)
      dPhi_ = dPhiNLBMap_5bit_256Max[dPhiBin];
    if (bits == 6)
      dPhi_ = dPhiNLBMap_6bit_256Max[dPhiBin];
    if (bits == 7)
      dPhi_ = dPhiNLBMap_7bit_256Max[dPhiBin];
  }

  else if (max==512)
  {
    if (bits == 7)
      dPhi_ = dPhiNLBMap_7bit_512Max[dPhiBin];
    if (bits == 8)
      dPhi_ = dPhiNLBMap_8bit_512Max[dPhiBin];
  }

  return (dPhi_);
}

int PtAssignmentEngineAux2016::getCLCT(int clct) const {
  int clct_ = 0;
  int sign_ = 1;

  switch (clct) {
  case 10: clct_ = 0; sign_ = -1; break;
  case  9: clct_ = 1; sign_ =  1; break;
  case  8: clct_ = 1; sign_ = -1; break;
  case  7: clct_ = 2; sign_ =  1; break;
  case  6: clct_ = 2; sign_ = -1; break;
  case  5: clct_ = 3; sign_ =  1; break;
  case  4: clct_ = 3; sign_ = -1; break;
  case  3: clct_ = 3; sign_ =  1; break;
  case  2: clct_ = 3; sign_ = -1; break;
  case  1: clct_ = 3; sign_ = -1; break;  //case  1: clct_ = 3; sign_ =  1; break;
  case  0: clct_ = 3; sign_ = -1; break;
  default: clct_ = 3; sign_ = -1; break;
  }
  return (sign_ * clct_);
}

int PtAssignmentEngineAux2016::getdTheta(int dTheta) const {
  int dTheta_ = 0;

  if (dTheta<=-3)
    dTheta_ = 0;
  else if (dTheta<=-2)
    dTheta_ = 1;
  else if (dTheta<=-1)
    dTheta_ = 2;
  else if (dTheta<=0)
    dTheta_ = 3;
  else if (dTheta<=1)
    dTheta_ = 4;
  else if (dTheta<=2)
    dTheta_ = 5;
  else if (dTheta<=3)
    dTheta_ = 6;
  else
    dTheta_ = 7;
  return (dTheta_);
}

int PtAssignmentEngineAux2016::getdEta(int dEta) const {
  int dEta_ = 0;

  if (dEta<=-5)
    dEta_ = 0;
  else if (dEta<=-2)
    dEta_ = 1;
  else if (dEta<=-1)
    dEta_ = 2;
  else if (dEta<=0)
    dEta_ = 3;
  else if (dEta<=1)
    dEta_ = 4;
  else if (dEta<=3)
    dEta_ = 5;
  else if (dEta<=6)
    dEta_ = 6;
  else
    dEta_ = 7;
  return (dEta_);
}

int PtAssignmentEngineAux2016::getEtaInt(float eta, int bits) const {
  eta = std::abs(eta);
  eta = (eta < 0.9) ? 0.9 : eta;
  bits = (bits > 5) ? 5 : bits;
  // encode 0.9<abs(eta)<1.6 in 5-bit (32 possible values)
  int etaInt = ((eta - 0.9)*32)/(1.6) - 0.5;
  int shift = 5 - bits;
  etaInt >>= shift;
  etaInt = (etaInt > 31) ? 31 : etaInt;
  return etaInt;
}

float PtAssignmentEngineAux2016::getEtaFromThetaInt(int thetaInt, int bits) const {
  thetaInt = (thetaInt > 127) ? 127 : thetaInt;  // 7-bit
  thetaInt = (thetaInt < 0) ? 0 : thetaInt;
  float theta = thetaInt;
  theta = (theta*0.2874016 + 8.5)*(3.14159265359/180);  // use this version to reproduce what happened when the pT LUT was made
  //theta = (theta*(45.0-8.5)/128. + 8.5) * M_PI/180.;
  float eta = -std::log(std::tan(theta/2));
  return eta;
}

float PtAssignmentEngineAux2016::getEtaFromEtaInt(int etaInt, int bits) const {
  etaInt = (etaInt > 31) ? 31 : etaInt;  // 5-bit
  etaInt = (etaInt < 0) ? 0 : etaInt;
  bits = (bits > 5) ? 5 : bits;
  int shift = 5 - bits;
  etaInt <<= shift;
  // decode 5-bit etaInt to 0.9<abs(eta)<1.6
  float eta = ((0.5 + etaInt)*1.6)/32 + 0.9;
  return eta;
}

float PtAssignmentEngineAux2016::getEtaFromBin(int etaBin, int bits) const {
  // For backward compatibility
  return getEtaFromEtaInt(etaBin, bits);
}

// front-rear LUTs
// [sector[0]][station 0-4][chamber id]
// chamber numbers start from 1, so add an extra low bit for invalid chamber = 0
static const int FRLUT[2][5] = {
  {0b0000000100100, 0b0000001011010, 0b0101010101010, 0b0010101010100, 0b0010101010100},
  {0b0000000100100, 0b0000001011010, 0b0111010100100, 0b0000101011010, 0b0000101011010}
};

int PtAssignmentEngineAux2016::getFRLUT(int sector, int station, int chamber) const {
  int bits = FRLUT[(sector-1)%2][station];
  bool isFront = bits & (1<<chamber);
  return isFront;
}
