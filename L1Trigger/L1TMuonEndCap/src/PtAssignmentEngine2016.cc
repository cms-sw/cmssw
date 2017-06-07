#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2016.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2016.h"

#include <cassert>
#include <iostream>
#include <sstream>

const PtAssignmentEngineAux2016& PtAssignmentEngine2016::aux() const {
  static const PtAssignmentEngineAux2016 instance;
  return instance;
}

float PtAssignmentEngine2016::scale_pt(const float pt, const int mode) const {

  // Scaling to achieve 90% efficency at any given L1 pT threshold
  // For 2016, was a flat scaling factor of 1.4

  float pt_scale = 1.4;

  return pt_scale;
}

float PtAssignmentEngine2016::unscale_pt(const float pt, const int mode) const {
  float pt_unscale = 1. / 1.4;
  return pt_unscale;
}



PtAssignmentEngine::address_t PtAssignmentEngine2016::calculate_address(const EMTFTrack& track) const {
  address_t address = 0;

  const EMTFPtLUT& ptlut_data = track.PtLUT();

  int mode_inv  = track.Mode_inv();
  int theta     = track.Theta_fp();
  theta >>= 2;  // truncate from 7-bit to 5-bit

  int dPhi12    = ptlut_data.delta_ph[0];
  int dPhi13    = ptlut_data.delta_ph[1];
  int dPhi14    = ptlut_data.delta_ph[2];
  int dPhi23    = ptlut_data.delta_ph[3];
  int dPhi24    = ptlut_data.delta_ph[4];
  int dPhi34    = ptlut_data.delta_ph[5];
  int dTheta12  = ptlut_data.delta_th[0];
  int dTheta13  = ptlut_data.delta_th[1];
  int dTheta14  = ptlut_data.delta_th[2];
  int dTheta23  = ptlut_data.delta_th[3];
  int dTheta24  = ptlut_data.delta_th[4];
  int dTheta34  = ptlut_data.delta_th[5];
  int FR1       = ptlut_data.fr      [0];
  int FR2       = ptlut_data.fr      [1];
  int FR3       = ptlut_data.fr      [2];
  int FR4       = ptlut_data.fr      [3];

  int sign12       = ptlut_data.sign_ph [0];
  int sign13       = ptlut_data.sign_ph [1];
  int sign14       = ptlut_data.sign_ph [2];
  int sign23       = ptlut_data.sign_ph [3];
  int sign24       = ptlut_data.sign_ph [4];
  int sign34       = ptlut_data.sign_ph [5];
  int dTheta12Sign = ptlut_data.sign_th [0];
  int dTheta13Sign = ptlut_data.sign_th [1];
  int dTheta14Sign = ptlut_data.sign_th [2];
  int dTheta23Sign = ptlut_data.sign_th [3];
  int dTheta24Sign = ptlut_data.sign_th [4];
  int dTheta34Sign = ptlut_data.sign_th [5];

  int CLCT1     = std::abs(aux().getCLCT(ptlut_data.cpattern[0]));
  int CLCT2     = std::abs(aux().getCLCT(ptlut_data.cpattern[1]));
  int CLCT3     = std::abs(aux().getCLCT(ptlut_data.cpattern[2]));
  int CLCT4     = std::abs(aux().getCLCT(ptlut_data.cpattern[3]));
  int CLCT1Sign = (aux().getCLCT(ptlut_data.cpattern[0]) > 0);
  int CLCT2Sign = (aux().getCLCT(ptlut_data.cpattern[1]) > 0);
  int CLCT3Sign = (aux().getCLCT(ptlut_data.cpattern[2]) > 0);
  int CLCT4Sign = (aux().getCLCT(ptlut_data.cpattern[3]) > 0);

  int CSCID1    = (ptlut_data.bt_vi[0] == 0 && ptlut_data.bt_vi[1] != 0) ?  ptlut_data.bt_ci[1]+16 : ptlut_data.bt_ci[0];
  int CSCID2    = ptlut_data.bt_ci[2];
  int CSCID3    = ptlut_data.bt_ci[3];
  int CSCID4    = ptlut_data.bt_ci[4];

  auto get_signed_int = [](int var, int sign) {
    return (sign == 1) ? (var * 1) : (var * -1);
  };

  dTheta12 = aux().getdTheta(get_signed_int(dTheta12, dTheta12Sign));
  dTheta13 = aux().getdTheta(get_signed_int(dTheta13, dTheta13Sign));
  dTheta14 = aux().getdTheta(get_signed_int(dTheta14, dTheta14Sign));
  dTheta23 = aux().getdTheta(get_signed_int(dTheta23, dTheta23Sign));
  dTheta24 = aux().getdTheta(get_signed_int(dTheta24, dTheta24Sign));
  dTheta34 = aux().getdTheta(get_signed_int(dTheta34, dTheta34Sign));

  bool use_FRLUT = true;
  if (use_FRLUT) {
    if (CSCID1 >= 16)
      FR1 = aux().getFRLUT(track.Sector(), 1, CSCID1-16);
    else
      FR1 = aux().getFRLUT(track.Sector(), 0, CSCID1);
    FR2 = aux().getFRLUT(track.Sector(), 2, CSCID2);
    FR3 = aux().getFRLUT(track.Sector(), 3, CSCID3);
    FR4 = aux().getFRLUT(track.Sector(), 4, CSCID4);
  }

  switch (mode_inv) {
  case 3:   // 1-2
    if (!bug9BitDPhi_)  dPhi12 = std::min(511, dPhi12);

    address |= (dPhi12      & ((1<<9)-1)) << (0);
    address |= (sign12      & ((1<<1)-1)) << (0+9);
    address |= (dTheta12    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT2       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT2Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR1         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR2         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 5:   // 1-3
    if (!bug9BitDPhi_)  dPhi13 = std::min(511, dPhi13);

    address |= (dPhi13      & ((1<<9)-1)) << (0);
    address |= (sign13      & ((1<<1)-1)) << (0+9);
    address |= (dTheta13    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT3       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT3Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR1         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR3         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 9:   // 1-4
    if (!bug9BitDPhi_)  dPhi14 = std::min(511, dPhi14);

    address |= (dPhi14      & ((1<<9)-1)) << (0);
    address |= (sign14      & ((1<<1)-1)) << (0+9);
    address |= (dTheta14    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT4       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT4Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR1         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR4         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 6:   // 2-3
    if (!bug9BitDPhi_)  dPhi23 = std::min(511, dPhi23);

    address |= (dPhi23      & ((1<<9)-1)) << (0);
    address |= (sign23      & ((1<<1)-1)) << (0+9);
    address |= (dTheta23    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT2       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT2Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT3       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT3Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR2         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR3         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 10:  // 2-4
    if (!bug9BitDPhi_)  dPhi24 = std::min(511, dPhi24);

    address |= (dPhi24      & ((1<<9)-1)) << (0);
    address |= (sign24      & ((1<<1)-1)) << (0+9);
    address |= (dTheta24    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT2       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT2Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT4       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT4Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR2         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR4         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 12:  // 3-4
    if (!bug9BitDPhi_)  dPhi34 = std::min(511, dPhi34);

    address |= (dPhi34      & ((1<<9)-1)) << (0);
    address |= (sign34      & ((1<<1)-1)) << (0+9);
    address |= (dTheta34    & ((1<<3)-1)) << (0+9+1);
    address |= (CLCT3       & ((1<<2)-1)) << (0+9+1+3);
    address |= (CLCT3Sign   & ((1<<1)-1)) << (0+9+1+3+2);
    address |= (CLCT4       & ((1<<2)-1)) << (0+9+1+3+2+1);
    address |= (CLCT4Sign   & ((1<<1)-1)) << (0+9+1+3+2+1+2);
    address |= (FR3         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1);
    address |= (FR4         & ((1<<1)-1)) << (0+9+1+3+2+1+2+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+9+1+3+2+1+2+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+9+1+3+2+1+2+1+1+1+5);
    break;

  case 7:   // 1-2-3
    dPhi12 = aux().getNLBdPhiBin(dPhi12, 7, 512);
    dPhi23 = aux().getNLBdPhiBin(dPhi23, 5, 256);

    address |= (dPhi12      & ((1<<7)-1)) << (0);
    address |= (dPhi23      & ((1<<5)-1)) << (0+7);
    address |= (sign12      & ((1<<1)-1)) << (0+7+5);
    address |= (sign23      & ((1<<1)-1)) << (0+7+5+1);
    address |= (dTheta13    & ((1<<3)-1)) << (0+7+5+1+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+7+5+1+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+7+5+1+1+3+2);
    address |= (FR1         & ((1<<1)-1)) << (0+7+5+1+1+3+2+1);
    address |= (theta       & ((1<<5)-1)) << (0+7+5+1+1+3+2+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+7+5+1+1+3+2+1+1+5);
    break;

  case 11:  // 1-2-4
    dPhi12 = aux().getNLBdPhiBin(dPhi12, 7, 512);
    dPhi24 = aux().getNLBdPhiBin(dPhi24, 5, 256);

    address |= (dPhi12      & ((1<<7)-1)) << (0);
    address |= (dPhi24      & ((1<<5)-1)) << (0+7);
    address |= (sign12      & ((1<<1)-1)) << (0+7+5);
    address |= (sign24      & ((1<<1)-1)) << (0+7+5+1);
    address |= (dTheta14    & ((1<<3)-1)) << (0+7+5+1+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+7+5+1+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+7+5+1+1+3+2);
    address |= (FR1         & ((1<<1)-1)) << (0+7+5+1+1+3+2+1);
    address |= (theta       & ((1<<5)-1)) << (0+7+5+1+1+3+2+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+7+5+1+1+3+2+1+1+5);
    break;

  case 13:  // 1-3-4
    dPhi13 = aux().getNLBdPhiBin(dPhi13, 7, 512);
    dPhi34 = aux().getNLBdPhiBin(dPhi34, 5, 256);

    address |= (dPhi13      & ((1<<7)-1)) << (0);
    address |= (dPhi34      & ((1<<5)-1)) << (0+7);
    address |= (sign13      & ((1<<1)-1)) << (0+7+5);
    address |= (sign34      & ((1<<1)-1)) << (0+7+5+1);
    address |= (dTheta14    & ((1<<3)-1)) << (0+7+5+1+1);
    address |= (CLCT1       & ((1<<2)-1)) << (0+7+5+1+1+3);
    address |= (CLCT1Sign   & ((1<<1)-1)) << (0+7+5+1+1+3+2);
    address |= (FR1         & ((1<<1)-1)) << (0+7+5+1+1+3+2+1);
    address |= (theta       & ((1<<5)-1)) << (0+7+5+1+1+3+2+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+7+5+1+1+3+2+1+1+5);
    break;

  case 14:  // 2-3-4
    dPhi23 = aux().getNLBdPhiBin(dPhi23, 7, 512);
    dPhi34 = aux().getNLBdPhiBin(dPhi34, 6, 256);

    address |= (dPhi23      & ((1<<7)-1)) << (0);
    address |= (dPhi34      & ((1<<6)-1)) << (0+7);
    address |= (sign23      & ((1<<1)-1)) << (0+7+6);
    address |= (sign34      & ((1<<1)-1)) << (0+7+6+1);
    address |= (dTheta24    & ((1<<3)-1)) << (0+7+6+1+1);
    address |= (CLCT2       & ((1<<2)-1)) << (0+7+6+1+1+3);
    address |= (CLCT2Sign   & ((1<<1)-1)) << (0+7+6+1+1+3+2);
    address |= (theta       & ((1<<5)-1)) << (0+7+6+1+1+3+2+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+7+6+1+1+3+2+1+5);
    break;

  case 15:  // 1-2-3-4
    // Set sign23 and sign34 relative to sign12
    if (!sign12) {
      sign12 = !sign12;
      sign23 = !sign23;
      sign34 = !sign34;
    }

    dPhi12 = aux().getNLBdPhiBin(dPhi12, 7, 512);
    dPhi23 = aux().getNLBdPhiBin(dPhi23, 5, 256);
    dPhi34 = aux().getNLBdPhiBin(dPhi34, 6, 256);

    address |= (dPhi12      & ((1<<7)-1)) << (0);
    address |= (dPhi23      & ((1<<5)-1)) << (0+7);
    address |= (dPhi34      & ((1<<6)-1)) << (0+7+5);
    address |= (sign23      & ((1<<1)-1)) << (0+7+5+6);
    address |= (sign34      & ((1<<1)-1)) << (0+7+5+6+1);
    address |= (FR1         & ((1<<1)-1)) << (0+7+5+6+1+1);
    address |= (theta       & ((1<<5)-1)) << (0+7+5+6+1+1+1);
    address |= (mode_inv    & ((1<<4)-1)) << (0+7+5+6+1+1+1+5);
    break;

  default:
    break;
  }

  return address;
}

float PtAssignmentEngine2016::calculate_pt_xml(const address_t& address) const {
  float pt = 0.;

  if (address == 0)  // invalid address
    return -1;  // return pt;

  int mode_inv = (address >> (30-4)) & ((1<<4)-1);

  auto contain = [](const std::vector<int>& vec, int elem) {
    return (std::find(vec.begin(), vec.end(), elem) != vec.end());
  };

  bool is_good_mode = contain(allowedModes_, mode_inv);

  if (!is_good_mode)  // invalid mode
    return -1;  // return pt;

  int dPhi12    = -999;
  int dPhi13    = -999;
  int dPhi14    = -999;
  int dPhi23    = -999;
  int dPhi24    = -999;
  int dPhi34    = -999;
  int dTheta12  = -999;
  int dTheta13  = -999;
  int dTheta14  = -999;
  int dTheta23  = -999;
  int dTheta24  = -999;
  int dTheta34  = -999;
  int CLCT1     = -999;
  int CLCT2     = -999;
  int CLCT3     = -999;
  int CLCT4     = -999;
  int CSCID1    = -999;
  int CSCID2    = -999;
  int CSCID3    = -999;
  int CSCID4    = -999;
  int FR1       = -999;
  int FR2       = -999;
  int FR3       = -999;
  int FR4       = -999;

  int sign12    = 1;
  int sign13    = 1;
  int sign14    = 1;
  int sign23    = 1;
  int sign24    = 1;
  int sign34    = 1;

  int CLCT1Sign = 1;
  int CLCT2Sign = 1;
  int CLCT3Sign = 1;
  int CLCT4Sign = 1;

  int theta     = 0;

  switch (mode_inv) {
  case 3:   // 1-2
    dPhi12    = (address >> (0))                    & ((1<<9)-1);
    sign12    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta12  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT1     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT1Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT2     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT2Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR1       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR2       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 5:   // 1-3
    dPhi13    = (address >> (0))                    & ((1<<9)-1);
    sign13    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta13  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT1     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT1Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT3     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT3Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR1       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR3       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 9:   // 1-4
    dPhi14    = (address >> (0))                    & ((1<<9)-1);
    sign14    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta14  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT1     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT1Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT4     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT4Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR1       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR4       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 6:   // 2-3
    dPhi23    = (address >> (0))                    & ((1<<9)-1);
    sign23    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta23  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT2     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT2Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT3     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT3Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR2       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR3       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 10:  // 2-4
    dPhi24    = (address >> (0))                    & ((1<<9)-1);
    sign24    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta24  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT2     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT2Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT4     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT4Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR2       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR4       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 12:  // 3-4
    dPhi34    = (address >> (0))                    & ((1<<9)-1);
    sign34    = (address >> (0+9))                  & ((1<<1)-1);
    dTheta34  = (address >> (0+9+1))                & ((1<<3)-1);
    CLCT3     = (address >> (0+9+1+3))              & ((1<<2)-1);
    CLCT3Sign = (address >> (0+9+1+3+2))            & ((1<<1)-1);
    CLCT4     = (address >> (0+9+1+3+2+1))          & ((1<<2)-1);
    CLCT4Sign = (address >> (0+9+1+3+2+1+2))        & ((1<<1)-1);
    FR3       = (address >> (0+9+1+3+2+1+2+1))      & ((1<<1)-1);
    FR4       = (address >> (0+9+1+3+2+1+2+1+1))    & ((1<<1)-1);
    theta     = (address >> (0+9+1+3+2+1+2+1+1+1))  & ((1<<5)-1);
    break;

  case 7:   // 1-2-3
    dPhi12    = (address >> (0))                    & ((1<<7)-1);
    dPhi23    = (address >> (0+7))                  & ((1<<5)-1);
    sign12    = (address >> (0+7+5))                & ((1<<1)-1);
    sign23    = (address >> (0+7+5+1))              & ((1<<1)-1);
    dTheta13  = (address >> (0+7+5+1+1))            & ((1<<3)-1);
    CLCT1     = (address >> (0+7+5+1+1+3))          & ((1<<2)-1);
    CLCT1Sign = (address >> (0+7+5+1+1+3+2))        & ((1<<1)-1);
    FR1       = (address >> (0+7+5+1+1+3+2+1))      & ((1<<1)-1);
    theta     = (address >> (0+7+5+1+1+3+2+1+1))    & ((1<<5)-1);

    dPhi12 = aux().getdPhiFromBin(dPhi12, 7, 512);
    dPhi23 = aux().getdPhiFromBin(dPhi23, 5, 256);
    break;

  case 11:  // 1-2-4
    dPhi12    = (address >> (0))                    & ((1<<7)-1);
    dPhi24    = (address >> (0+7))                  & ((1<<5)-1);
    sign12    = (address >> (0+7+5))                & ((1<<1)-1);
    sign24    = (address >> (0+7+5+1))              & ((1<<1)-1);
    dTheta14  = (address >> (0+7+5+1+1))            & ((1<<3)-1);
    CLCT1     = (address >> (0+7+5+1+1+3))          & ((1<<2)-1);
    CLCT1Sign = (address >> (0+7+5+1+1+3+2))        & ((1<<1)-1);
    FR1       = (address >> (0+7+5+1+1+3+2+1))      & ((1<<1)-1);
    theta     = (address >> (0+7+5+1+1+3+2+1+1))    & ((1<<5)-1);

    dPhi12 = aux().getdPhiFromBin(dPhi12, 7, 512);
    dPhi24 = aux().getdPhiFromBin(dPhi24, 5, 256);
    break;

  case 13:  // 1-3-4
    dPhi13    = (address >> (0))                    & ((1<<7)-1);
    dPhi34    = (address >> (0+7))                  & ((1<<5)-1);
    sign13    = (address >> (0+7+5))                & ((1<<1)-1);
    sign34    = (address >> (0+7+5+1))              & ((1<<1)-1);
    dTheta14  = (address >> (0+7+5+1+1))            & ((1<<3)-1);
    CLCT1     = (address >> (0+7+5+1+1+3))          & ((1<<2)-1);
    CLCT1Sign = (address >> (0+7+5+1+1+3+2))        & ((1<<1)-1);
    FR1       = (address >> (0+7+5+1+1+3+2+1))      & ((1<<1)-1);
    theta     = (address >> (0+7+5+1+1+3+2+1+1))    & ((1<<5)-1);

    dPhi13 = aux().getdPhiFromBin(dPhi13, 7, 512);
    dPhi34 = aux().getdPhiFromBin(dPhi34, 5, 256);
    break;

  case 14:  // 2-3-4
    dPhi23    = (address >> (0))                    & ((1<<7)-1);
    dPhi34    = (address >> (0+7))                  & ((1<<6)-1);
    sign23    = (address >> (0+7+6))                & ((1<<1)-1);
    sign34    = (address >> (0+7+6+1))              & ((1<<1)-1);
    dTheta24  = (address >> (0+7+6+1+1))            & ((1<<3)-1);
    CLCT2     = (address >> (0+7+6+1+1+3))          & ((1<<2)-1);
    CLCT2Sign = (address >> (0+7+6+1+1+3+2))        & ((1<<1)-1);
    theta     = (address >> (0+7+6+1+1+3+2+1))      & ((1<<5)-1);

    dPhi23 = aux().getdPhiFromBin(dPhi23, 7, 512);
    dPhi34 = aux().getdPhiFromBin(dPhi34, 6, 256);
    break;

  case 15:  // 1-2-3-4
    dPhi12    = (address >> (0))                    & ((1<<7)-1);
    dPhi23    = (address >> (0+7))                  & ((1<<5)-1);
    dPhi34    = (address >> (0+7+5))                & ((1<<6)-1);
    sign23    = (address >> (0+7+5+6))              & ((1<<1)-1);
    sign34    = (address >> (0+7+5+6+1))            & ((1<<1)-1);
    FR1       = (address >> (0+7+5+6+1+1))          & ((1<<1)-1);
    theta     = (address >> (0+7+5+6+1+1+1))        & ((1<<5)-1);

    dPhi12 = aux().getdPhiFromBin(dPhi12, 7, 512);
    dPhi23 = aux().getdPhiFromBin(dPhi23, 5, 256);
    dPhi34 = aux().getdPhiFromBin(dPhi34, 6, 256);
    break;

  default:
    break;
  }


  auto get_signed_int = [](int var, int sign) {
    return (sign == 1) ? (var * 1) : (var * -1);
  };

  dPhi12 = get_signed_int(dPhi12, sign12);
  dPhi13 = get_signed_int(dPhi13, sign13);
  dPhi14 = get_signed_int(dPhi14, sign14);
  dPhi23 = get_signed_int(dPhi23, sign23);
  dPhi24 = get_signed_int(dPhi24, sign24);
  dPhi34 = get_signed_int(dPhi34, sign34);

  CLCT1  = get_signed_int(CLCT1, CLCT1Sign);
  CLCT2  = get_signed_int(CLCT2, CLCT2Sign);
  CLCT3  = get_signed_int(CLCT3, CLCT3Sign);
  CLCT4  = get_signed_int(CLCT4, CLCT4Sign);

  theta <<= 2;
  float eta = aux().getEtaFromThetaInt(theta, 5);

  bool use_lossy_eta = true;
  if (use_lossy_eta) {
    int etaInt = aux().getEtaInt(eta, 5);
    etaInt &= ((1<<5)-1);
    eta = aux().getEtaFromEtaInt(etaInt, 5);
  }


  // First fix to recover high pT muons with 3 hits in a line and one displaced hit
  // Done by re-writing a few addresses in the original LUT, according to the following logic
  // Implemented in FW 26.07.16, as of run 2774278 / fill 5119
  if (fixMode15HighPt_) {
    if (mode_inv == 15) {  // 1-2-3-4
      bool st2_off = false;
      bool st3_off = false;
      bool st4_off = false;

      dPhi13 = dPhi12 + dPhi23;
      dPhi14 = dPhi13 + dPhi34;
      dPhi24 = dPhi23 + dPhi34;

      int sum_st1 = abs( dPhi12 + dPhi13 + dPhi14);
      int sum_st2 = abs(-dPhi12 + dPhi23 + dPhi24);
      int sum_st3 = abs(-dPhi13 - dPhi23 + dPhi34);
      int sum_st4 = abs(-dPhi14 - dPhi24 - dPhi34);

      // Detect outliers
      if (sum_st2 > sum_st1 && sum_st2 > sum_st3 && sum_st2 > sum_st4) st2_off = true;
      if (sum_st3 > sum_st1 && sum_st3 > sum_st2 && sum_st3 > sum_st4) st3_off = true;
      if (sum_st4 > sum_st1 && sum_st4 > sum_st2 && sum_st4 > sum_st3) st4_off = true;

      // Recover outliers
      if (st2_off) {
        if (
            (abs(dPhi12) > 9 || abs(dPhi23) > 9 || abs(dPhi24) > 9) &&
            (abs(dPhi13) < 10 && abs(dPhi14) < 10 && abs(dPhi34) < 10)
        ) {
          dPhi12 = dPhi13 / 2;
          dPhi23 = dPhi13 / 2;
        }
      }
      if (st3_off) {
        if (
            (abs(dPhi13) > 9 || abs(dPhi23) > 9 || abs(dPhi34) > 9) &&
            (abs(dPhi12) < 10 && abs(dPhi14) < 10 && abs(dPhi24) < 10)
        ) {
          dPhi23 = dPhi24 / 2;
          dPhi34 = dPhi24 / 2;
        }
      }
      if (st4_off) {
        if (
            (abs(dPhi14) > 9 || abs(dPhi24) > 9 || abs(dPhi34) > 9 ) &&
            (abs(dPhi12) < 10 && abs(dPhi13) < 10 && abs(dPhi23) < 10)
        ) {
          if (abs(dPhi13) < abs(dPhi23))
            dPhi34 = dPhi13;
          else
            dPhi34 = dPhi23;
        }
      }

      // Set sign23 and sign34 relative to sign12
      //sign12 = (dPhi12 >= 0);
      //sign23 = (dPhi23 >= 0);
      //sign34 = (dPhi34 >= 0);
      //if (!sign12) {
      //  sign12 = !sign12;
      //  sign23 = !sign23;
      //  sign34 = !sign34;
      //}

      sign12 = 1;
      if (dPhi12 > 0) {
        sign23 = (dPhi23 > 0) ? 1 : 0;
        sign34 = (dPhi34 > 0) ? 1 : 0;
      } else {
        sign23 = (dPhi23 < 0) ? 1 : 0;
        sign34 = (dPhi34 < 0) ? 1 : 0;
      }

      dPhi12 = get_signed_int(abs(dPhi12), sign12);
      dPhi23 = get_signed_int(abs(dPhi23), sign23);
      dPhi34 = get_signed_int(abs(dPhi34), sign34);

    }  // end if mode_inv == 15
  }

  // Inherit some bugs
  if (bugMode7CLCT_) {
    if (mode_inv == 14) {  // 2-3-4
      int bugged_CLCT2;
      address_t bugged_address;

      //CLCT2     = (address >> (0+7+6+1+1+3))          & ((1<<2)-1);
      //address |= (CLCT2       & ((1<<2)-1)) << (0+7+6+1+1+3);

      bugged_CLCT2    = (address >> (0+7+5+1+1+3))          & ((1<<2)-1);  // bad
      bugged_address  = address & ~(((1<<2)-1) << (0+7+6+1+1+3));  // clear bits
      bugged_address |= (bugged_CLCT2 & ((1<<2)-1)) << (0+7+6+1+1+3);
      bugged_CLCT2    = (bugged_address >> (0+7+5+1+1+3))   & ((1<<2)-1);  // bad

      CLCT2  = bugged_CLCT2;
      CLCT2  = get_signed_int(CLCT2, CLCT2Sign);

    }  // end if mode_inv == 14
  }


  // Get pT from XML (forest)
  const int (*mode_variables)[6] = aux().getModeVariables();

  std::vector<int> variables = {
    dPhi12, dPhi13, dPhi14, dPhi23, dPhi24, dPhi34, dTheta12, dTheta13, dTheta14, dTheta23, dTheta24, dTheta34,
    CLCT1, CLCT2, CLCT3, CLCT4, CSCID1, CSCID2, CSCID3, CSCID4, FR1, FR2, FR3, FR4
  };

  std::vector<Double_t> tree_data;
  tree_data.push_back(1.0);
  tree_data.push_back(eta);

  for (int i=0; i<6; i++) {  // loop over 6 variables (or less)
    int mv = mode_variables[mode_inv-3][i];
    if (mv != -999) {
      int v = variables.at(mv);
      if (!(mode_inv == 13 && i == 3)) {  // somehow this uses CSCID1
        assert(v != -999);
      }
      tree_data.push_back(v);
    }
  }

  if (verbose_ > 2) {
    std::cout << "mode_inv: " << mode_inv << " variables: ";
    for (const auto& v: tree_data)
      std::cout << v << " ";
    std::cout << std::endl;
  }

  auto tree_event = std::make_unique<emtf::Event>();
  tree_event->predictedValue = 0;  // must explicitly initialize
  tree_event->data = tree_data;

  // forests_.at(mode_inv).predictEvent(tree_event.get(), 64);
  emtf::Forest &forest = const_cast<emtf::Forest&>(forests_.at(mode_inv));
  forest.predictEvent(tree_event.get(), 64);

  float tmp_pt = tree_event->predictedValue;  // is actually 1/pT

  if (verbose_ > 1) {
    std::cout << "mode_inv: " << mode_inv << " 1/pT: " << tmp_pt << std::endl;
    std::cout << "dPhi12: " << dPhi12 << " dPhi13: " << dPhi13 << " dPhi14: " << dPhi14
        << " dPhi23: " << dPhi23 << " dPhi24: " << dPhi24 << " dPhi34: " << dPhi34 << std::endl;
    std::cout << "dTheta12: " << dTheta12 << " dTheta13: " << dTheta13 << " dTheta14: " << dTheta14
        << " dTheta23: " << dTheta23 << " dTheta24: " << dTheta24 << " dTheta34: " << dTheta34 << std::endl;
    std::cout << "CLCT1: " << CLCT1 << " CLCT2: " << CLCT2 << " CLCT3: " << CLCT3 << " CLCT4: " << CLCT4 << std::endl;
    std::cout << "CSCID1: " << CSCID1 << " CSCID2: " << CSCID2 << " CSCID3: " << CSCID3 << " CSCID4: " << CSCID4 << std::endl;
    std::cout << "FR1: " << FR1 << " FR2: " << FR2 << " FR3: " << FR3 << " FR4: " << FR4 << std::endl;
  }

  if (bugNegPt_) {
    pt = (tmp_pt == 0) ? tmp_pt : 1.0/tmp_pt;
    if (pt<0.0)   pt = 1.0;
    if (pt>200.0) pt = 200.0;

  } else {
    if (tmp_pt < 0.0)  tmp_pt = 1.0/7000;
    pt = (tmp_pt == 0) ? tmp_pt : 1.0/tmp_pt;
  }

  assert(pt > 0);
  return pt;
}


// Not implemented for 2016
float PtAssignmentEngine2016::calculate_pt_xml(const EMTFTrack& track) const {
  float pt = 0.;

  return pt;
} 
