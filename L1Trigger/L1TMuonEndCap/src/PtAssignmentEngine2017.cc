#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.h"

#include <iostream>
#include <sstream>

const PtAssignmentEngineAux2017& PtAssignmentEngine2017::aux() const {
  static const PtAssignmentEngineAux2017
      instance;  // KK: arguable design solution, but const qualifier makes it thread-safe anyway
  return instance;
}

float PtAssignmentEngine2017::scale_pt(const float pt, const int mode) const {
  emtf_assert(ptLUTVersion_ >= 6);

  float pt_xml = -99;
  float pt_scale = -99;

  // Scaling to achieve 90% efficency at any given L1 pT threshold
  // For now, a linear scaling based on SingleMu-quality (modes 11, 13, 14, 15), CSC+RPC tracks
  // Should maybe scale each mode differently in the future - AWB 31.05.17

  // TRG       = (1.2 + 0.015*TRG) * XML
  // TRG       = 1.2*XML / (1 - 0.015*XML)
  // TRG / XML = 1.2 / (1 - 0.015*XML)

  if (ptLUTVersion_ >= 8) {  // First "physics" LUTs for 2022, will be deployed in June 2022
    pt_xml = fmin(20., pt);  // Maximum scale set by muons with XML pT = 20 GeV (scaled pT ~32 GeV)
    pt_scale = 1.13 / (1 - 0.015 * pt_xml);
  } else if (ptLUTVersion_ >= 6) {  // First "physics" LUTs for 2017, deployed June 7
    pt_xml = fmin(20., pt);         // Maximum scale set by muons with XML pT = 20 GeV (scaled pT ~35 GeV)
    pt_scale = 1.2 / (1 - 0.015 * pt_xml);
  }

  return pt_scale;
}

float PtAssignmentEngine2017::unscale_pt(const float pt, const int mode) const {
  emtf_assert(ptLUTVersion_ >= 6);

  float pt_unscale = -99;

  if (ptLUTVersion_ >= 8) {  // First "physics" LUTs for 2022, will be deployed in June 2022
    pt_unscale = 1 / (1.13 + 0.015 * pt);
    pt_unscale = fmax(pt_unscale, (1 - 0.015 * 20) / 1.13);
  } else if (ptLUTVersion_ >= 6) {  // First "physics" LUTs for 2017, deployed June 7
    pt_unscale = 1 / (1.2 + 0.015 * pt);
    pt_unscale = fmax(pt_unscale, (1 - 0.015 * 20) / 1.2);
  }

  return pt_unscale;
}

PtAssignmentEngine::address_t PtAssignmentEngine2017::calculate_address(const EMTFTrack& track) const {
  address_t address = 0;

  EMTFPtLUT data = track.PtLUT();

  int mode = track.Mode();
  int theta = track.Theta_fp();
  int endcap = track.Endcap();
  int nHits = (mode / 8) + ((mode % 8) / 4) + ((mode % 4) / 2) + ((mode % 2) / 1);
  emtf_assert(nHits > 1 && nHits < 5);

  // 'A' is first station in the track, 'B' the second, etc.
  int mode_ID = -1;
  int iA = -1, iB = -1, iC = -1, iD = -1;
  int iAB, iAC, iAD, iBC, iCD;

  switch (mode) {  // Indices for quantities by station or station pair
    case 15:
      mode_ID = 0b1;
      iA = 0;
      iB = 1;
      iC = 2;
      iD = 3;
      break;
    case 14:
      mode_ID = 0b11;
      iA = 0;
      iB = 1;
      iC = 2;
      break;
    case 13:
      mode_ID = 0b10;
      iA = 0;
      iB = 1;
      iC = 3;
      break;
    case 11:
      mode_ID = 0b01;
      iA = 0;
      iB = 2;
      iC = 3;
      break;
    case 7:
      mode_ID = 0b1;
      iA = 1;
      iB = 2;
      iC = 3;
      break;
    case 12:
      mode_ID = 0b111;
      iA = 0;
      iB = 1;
      break;
    case 10:
      mode_ID = 0b110;
      iA = 0;
      iB = 2;
      break;
    case 9:
      mode_ID = 0b101;
      iA = 0;
      iB = 3;
      break;
    case 6:
      mode_ID = 0b100;
      iA = 1;
      iB = 2;
      break;
    case 5:
      mode_ID = 0b011;
      iA = 1;
      iB = 3;
      break;
    case 3:
      mode_ID = 0b010;
      iA = 2;
      iB = 3;
      break;
    default:
      break;
  }
  iAB = (iA >= 0 && iB >= 0) ? iA + iB - (iA == 0) : -1;
  iAC = (iA >= 0 && iC >= 0) ? iA + iC - (iA == 0) : -1;
  iAD = (iA >= 0 && iD >= 0) ? 2 : -1;
  iBC = (iB >= 0 && iC >= 0) ? iB + iC : -1;
  iCD = (iC >= 0 && iD >= 0) ? 5 : -1;

  // Fill variable info from pT LUT data
  int st1_ring2 = data.st1_ring2;
  if (nHits == 4) {
    int dPhiAB = data.delta_ph[iAB];
    int dPhiBC = data.delta_ph[iBC];
    int dPhiCD = data.delta_ph[iCD];
    int sPhiAB = data.sign_ph[iAB];
    int sPhiBC = (data.sign_ph[iBC] == sPhiAB);
    int sPhiCD = (data.sign_ph[iCD] == sPhiAB);
    int dTheta = data.delta_th[iAD] * (data.sign_th[iAD] ? 1 : -1);
    int frA = data.fr[iA];
    int clctA = data.cpattern[iA];
    int clctB = data.cpattern[iB];
    int clctC = data.cpattern[iC];
    int clctD = data.cpattern[iD];

    // Convert variables to words for pT LUT address
    dPhiAB = aux().getNLBdPhiBin(dPhiAB, 7, 512);
    dPhiBC = aux().getNLBdPhiBin(dPhiBC, 5, 256);
    dPhiCD = aux().getNLBdPhiBin(dPhiCD, 4, 256);
    dTheta = aux().getdTheta(dTheta, 2);
    // Combines track theta, stations with RPC hits, and station 1 bend information
    int mode15_8b = aux().get8bMode15(theta, st1_ring2, endcap, (sPhiAB == 1 ? 1 : -1), clctA, clctB, clctC, clctD);

    // Form the pT LUT address
    address |= (dPhiAB & ((1 << 7) - 1)) << (0);
    address |= (dPhiBC & ((1 << 5) - 1)) << (0 + 7);
    address |= (dPhiCD & ((1 << 4) - 1)) << (0 + 7 + 5);
    address |= (sPhiBC & ((1 << 1) - 1)) << (0 + 7 + 5 + 4);
    address |= (sPhiCD & ((1 << 1) - 1)) << (0 + 7 + 5 + 4 + 1);
    address |= (dTheta & ((1 << 2) - 1)) << (0 + 7 + 5 + 4 + 1 + 1);
    address |= (frA & ((1 << 1) - 1)) << (0 + 7 + 5 + 4 + 1 + 1 + 2);
    address |= (mode15_8b & ((1 << 8) - 1)) << (0 + 7 + 5 + 4 + 1 + 1 + 2 + 1);
    address |= (mode_ID & ((1 << 1) - 1)) << (0 + 7 + 5 + 4 + 1 + 1 + 2 + 1 + 8);
    emtf_assert(address < pow(2, 30) && address >= pow(2, 29));
  } else if (nHits == 3) {
    int dPhiAB = data.delta_ph[iAB];
    int dPhiBC = data.delta_ph[iBC];
    int sPhiAB = data.sign_ph[iAB];
    int sPhiBC = (data.sign_ph[iBC] == sPhiAB);
    int dTheta = data.delta_th[iAC] * (data.sign_th[iAC] ? 1 : -1);
    int frA = data.fr[iA];
    int frB = data.fr[iB];
    int clctA = data.cpattern[iA];
    int clctB = data.cpattern[iB];
    int clctC = data.cpattern[iC];

    // Convert variables to words for pT LUT address
    dPhiAB = aux().getNLBdPhiBin(dPhiAB, 7, 512);
    dPhiBC = aux().getNLBdPhiBin(dPhiBC, 5, 256);
    dTheta = aux().getdTheta(dTheta, 3);
    // Identifies which stations have RPC hits in 3-station tracks
    int rpc_2b = aux().get2bRPC(clctA, clctB, clctC);  // Have to use un-compressed CLCT words
    clctA = aux().getCLCT(clctA, endcap, (sPhiAB == 1 ? 1 : -1), 2);
    theta = aux().getTheta(theta, st1_ring2, 5);

    // Form the pT LUT address
    address |= (dPhiAB & ((1 << 7) - 1)) << (0);
    address |= (dPhiBC & ((1 << 5) - 1)) << (0 + 7);
    address |= (sPhiBC & ((1 << 1) - 1)) << (0 + 7 + 5);
    address |= (dTheta & ((1 << 3) - 1)) << (0 + 7 + 5 + 1);
    address |= (frA & ((1 << 1) - 1)) << (0 + 7 + 5 + 1 + 3);
    int bit = 0;
    if (mode != 7) {
      address |= (frB & ((1 << 1) - 1)) << (0 + 7 + 5 + 1 + 3 + 1);
      bit = 1;
    }
    address |= (clctA & ((1 << 2) - 1)) << (0 + 7 + 5 + 1 + 3 + 1 + bit);
    address |= (rpc_2b & ((1 << 2) - 1)) << (0 + 7 + 5 + 1 + 3 + 1 + bit + 2);
    address |= (theta & ((1 << 5) - 1)) << (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2);
    if (mode != 7) {
      address |= (mode_ID & ((1 << 2) - 1)) << (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2 + 5);
      emtf_assert(address < pow(2, 29) && address >= pow(2, 27));
    } else {
      address |= (mode_ID & ((1 << 1) - 1)) << (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2 + 5);
      emtf_assert(address < pow(2, 27) && address >= pow(2, 26));
    }

  } else if (nHits == 2) {
    int dPhiAB = data.delta_ph[iAB];
    int sPhiAB = data.sign_ph[iAB];
    int dTheta = data.delta_th[iAB] * (data.sign_th[iAB] ? 1 : -1);
    int frA = data.fr[iA];
    int frB = data.fr[iB];
    int clctA = data.cpattern[iA];
    int clctB = data.cpattern[iB];

    // Convert variables to words for pT LUT address
    dPhiAB = aux().getNLBdPhiBin(dPhiAB, 7, 512);
    dTheta = aux().getdTheta(dTheta, 3);
    clctA = aux().getCLCT(clctA, endcap, (sPhiAB == 1 ? 1 : -1), 3);
    clctB = aux().getCLCT(clctB, endcap, (sPhiAB == 1 ? 1 : -1), 3);
    theta = aux().getTheta(theta, st1_ring2, 5);

    // Form the pT LUT address
    address |= (dPhiAB & ((1 << 7) - 1)) << (0);
    address |= (dTheta & ((1 << 3) - 1)) << (0 + 7);
    address |= (frA & ((1 << 1) - 1)) << (0 + 7 + 3);
    address |= (frB & ((1 << 1) - 1)) << (0 + 7 + 3 + 1);
    address |= (clctA & ((1 << 3) - 1)) << (0 + 7 + 3 + 1 + 1);
    address |= (clctB & ((1 << 3) - 1)) << (0 + 7 + 3 + 1 + 1 + 3);
    address |= (theta & ((1 << 5) - 1)) << (0 + 7 + 3 + 1 + 1 + 3 + 3);
    address |= (mode_ID & ((1 << 3) - 1)) << (0 + 7 + 3 + 1 + 1 + 3 + 3 + 5);
    emtf_assert(address < pow(2, 26) && address >= pow(2, 24));
  }
  return address;
}  // End function: PtAssignmentEngine2017::calculate_address()

// Calculate XML pT from address
float PtAssignmentEngine2017::calculate_pt_xml(const address_t& address) const {
  // std::cout << "Inside calculate_pt_xml, examining address: ";
  // for (int j = 0; j < 30; j++) {
  //   std::cout << ((address >> (29 - j)) & 0x1);
  //   if ((j % 5) == 4) std::cout << "  ";
  // }
  // std::cout << std::endl;

  float pt_xml = 0.;
  int nHits = -1, mode = -1;

  emtf_assert(address < pow(2, 30));
  if (address >= pow(2, 29)) {
    nHits = 4;
    mode = 15;
  } else if (address >= pow(2, 27)) {
    nHits = 3;
  } else if (address >= pow(2, 26)) {
    nHits = 3;
    mode = 7;
  } else if (address >= pow(2, 24)) {
    nHits = 2;
  } else
    return pt_xml;

  // Variables to unpack from the pT LUT address
  int mode_ID, theta, dTheta;
  int dPhiAB, dPhiBC = -1, dPhiCD = -1;
  int sPhiAB, sPhiBC = -1, sPhiCD = -1;
  int frA, frB = -1;
  int clctA, clctB = -1;
  int rpcA, rpcB, rpcC, rpcD;
  int endcap = 1;
  sPhiAB = 1;          // Assume positive endcap and dPhiAB for unpacking CLCT bend
  int mode15_8b = -1;  // Combines track theta, stations with RPC hits, and station 1 bend information
  int rpc_2b = -1;     // Identifies which stations have RPC hits in 3-station tracks

  // Unpack variable words from the pT LUT address
  if (nHits == 4) {
    dPhiAB = (address >> (0) & ((1 << 7) - 1));
    dPhiBC = (address >> (0 + 7) & ((1 << 5) - 1));
    dPhiCD = (address >> (0 + 7 + 5) & ((1 << 4) - 1));
    sPhiBC = (address >> (0 + 7 + 5 + 4) & ((1 << 1) - 1));
    sPhiCD = (address >> (0 + 7 + 5 + 4 + 1) & ((1 << 1) - 1));
    dTheta = (address >> (0 + 7 + 5 + 4 + 1 + 1) & ((1 << 2) - 1));
    frA = (address >> (0 + 7 + 5 + 4 + 1 + 1 + 2) & ((1 << 1) - 1));
    mode15_8b = (address >> (0 + 7 + 5 + 4 + 1 + 1 + 2 + 1) & ((1 << 8) - 1));
    mode_ID = (address >> (0 + 7 + 5 + 4 + 1 + 1 + 2 + 1 + 8) & ((1 << 1) - 1));
    emtf_assert(address < pow(2, 30));
  } else if (nHits == 3) {
    dPhiAB = (address >> (0) & ((1 << 7) - 1));
    dPhiBC = (address >> (0 + 7) & ((1 << 5) - 1));
    sPhiBC = (address >> (0 + 7 + 5) & ((1 << 1) - 1));
    dTheta = (address >> (0 + 7 + 5 + 1) & ((1 << 3) - 1));
    frA = (address >> (0 + 7 + 5 + 1 + 3) & ((1 << 1) - 1));
    int bit = 0;
    if (mode != 7) {
      frB = (address >> (0 + 7 + 5 + 1 + 3 + 1) & ((1 << 1) - 1));
      bit = 1;
    }
    clctA = (address >> (0 + 7 + 5 + 1 + 3 + 1 + bit) & ((1 << 2) - 1));
    rpc_2b = (address >> (0 + 7 + 5 + 1 + 3 + 1 + bit + 2) & ((1 << 2) - 1));
    theta = (address >> (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2) & ((1 << 5) - 1));
    if (mode != 7) {
      mode_ID = (address >> (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2 + 5) & ((1 << 2) - 1));
      emtf_assert(address < pow(2, 29));
    } else {
      mode_ID = (address >> (0 + 7 + 5 + 1 + 3 + 1 + bit + 2 + 2 + 5) & ((1 << 1) - 1));
      emtf_assert(address < pow(2, 27));
    }
  } else if (nHits == 2) {
    dPhiAB = (address >> (0) & ((1 << 7) - 1));
    dTheta = (address >> (0 + 7) & ((1 << 3) - 1));
    frA = (address >> (0 + 7 + 3) & ((1 << 1) - 1));
    frB = (address >> (0 + 7 + 3 + 1) & ((1 << 1) - 1));
    clctA = (address >> (0 + 7 + 3 + 1 + 1) & ((1 << 3) - 1));
    clctB = (address >> (0 + 7 + 3 + 1 + 1 + 3) & ((1 << 3) - 1));
    theta = (address >> (0 + 7 + 3 + 1 + 1 + 3 + 3) & ((1 << 5) - 1));
    mode_ID = (address >> (0 + 7 + 3 + 1 + 1 + 3 + 3 + 5) & ((1 << 3) - 1));
    emtf_assert(address < pow(2, 26));
  }

  // Infer track mode (and stations with hits) from mode_ID
  if (nHits == 3 && mode != 7) {
    switch (mode_ID) {
      case 0b11:
        mode = 14;
        break;
      case 0b10:
        mode = 13;
        break;
      case 0b01:
        mode = 11;
        break;
      default:
        break;
    }
  } else if (nHits == 2) {
    switch (mode_ID) {
      case 0b111:
        mode = 12;
        break;
      case 0b110:
        mode = 10;
        break;
      case 0b101:
        mode = 9;
        break;
      case 0b100:
        mode = 6;
        break;
      case 0b011:
        mode = 5;
        break;
      case 0b010:
        mode = 3;
        break;
      default:
        break;
    }
  }

  emtf_assert(mode > 0);

  // Un-compress words from address
  // For most variables (e.g. theta, dTheta, CLCT) don't need to unpack, since compressed version was used in training
  int St1_ring2 = -1;
  if (nHits == 4) {
    dPhiAB = aux().getdPhiFromBin(dPhiAB, 7, 512);
    dPhiBC = aux().getdPhiFromBin(dPhiBC, 5, 256) * (sPhiBC == 1 ? 1 : -1);
    dPhiCD = aux().getdPhiFromBin(dPhiCD, 4, 256) * (sPhiCD == 1 ? 1 : -1);
    aux().unpack8bMode15(mode15_8b, theta, St1_ring2, endcap, (sPhiAB == 1 ? 1 : -1), clctA, rpcA, rpcB, rpcC, rpcD);

    // // Check bit-wise compression / de-compression
    // emtf_assert( dTheta == aux().getdTheta( aux().unpackdTheta( dTheta, 2), 2) );
  } else if (nHits == 3) {
    dPhiAB = aux().getdPhiFromBin(dPhiAB, 7, 512);
    dPhiBC = aux().getdPhiFromBin(dPhiBC, 5, 256) * (sPhiBC == 1 ? 1 : -1);
    St1_ring2 = aux().unpackSt1Ring2(theta, 5);
    aux().unpack2bRPC(rpc_2b, rpcA, rpcB, rpcC);

    // // Check bit-wise compression / de-compression
    // emtf_assert( dTheta == aux().getdTheta( aux().unpackdTheta( dTheta, 3), 3) );
    // emtf_assert( clctA  == aux().getCLCT( aux().unpackCLCT( clctA, endcap, (sPhiAB == 1 ? 1 : -1), 2),
    //                               endcap, (sPhiAB == 1 ? 1 : -1), 2) );
    // int theta_unp = theta;
    // aux().unpackTheta( theta_unp, St1_ring2, 5 );
    // emtf_assert( theta == aux().getTheta(theta_unp, St1_ring2, 5) );
  } else if (nHits == 2) {
    dPhiAB = aux().getdPhiFromBin(dPhiAB, 7, 512);
    St1_ring2 = aux().unpackSt1Ring2(theta, 5);

    // // Check bit-wise compression / de-compression
    // emtf_assert( dTheta == aux().getdTheta( aux().unpackdTheta( dTheta, 3), 3) );
    // emtf_assert( clctA  == aux().getCLCT( aux().unpackCLCT( clctA, endcap, (sPhiAB == 1 ? 1 : -1), 3),
    //                               endcap, (sPhiAB == 1 ? 1 : -1), 3) );
    // emtf_assert( clctB  == aux().getCLCT( aux().unpackCLCT( clctB, endcap, (sPhiAB == 1 ? 1 : -1), 3),
    //                               endcap, (sPhiAB == 1 ? 1 : -1), 3) );
    // int theta_unp = theta;
    // aux().unpackTheta( theta_unp, St1_ring2, 5 );
    // emtf_assert( theta == aux().getTheta(theta_unp, St1_ring2, 5) );
  }

  // Fill vectors of variables for XMLs
  // KK: sequence of variables here should exaclty match <Variables> block produced by TMVA
  std::vector<int> predictors;

  // Variables for input to XMLs
  int dPhiSum4, dPhiSum4A, dPhiSum3, dPhiSum3A, outStPhi;

  // Convert words into variables for XMLs
  if (nHits == 4) {
    predictors = {
        theta, St1_ring2, dPhiAB, dPhiBC, dPhiCD, dPhiAB + dPhiBC, dPhiAB + dPhiBC + dPhiCD, dPhiBC + dPhiCD, frA, clctA};

    aux().calcDeltaPhiSums(dPhiSum4,
                           dPhiSum4A,
                           dPhiSum3,
                           dPhiSum3A,
                           outStPhi,
                           dPhiAB,
                           dPhiAB + dPhiBC,
                           dPhiAB + dPhiBC + dPhiCD,
                           dPhiBC,
                           dPhiBC + dPhiCD,
                           dPhiCD);

    int tmp[10] = {dPhiSum4, dPhiSum4A, dPhiSum3, dPhiSum3A, outStPhi, dTheta, rpcA, rpcB, rpcC, rpcD};
    predictors.insert(predictors.end(), tmp, tmp + 10);
  } else if (nHits == 3) {
    if (mode == 14)
      predictors = {theta, St1_ring2, dPhiAB, dPhiBC, dPhiAB + dPhiBC, frA, frB, clctA, dTheta, rpcA, rpcB, rpcC};
    else if (mode == 13)
      predictors = {theta, St1_ring2, dPhiAB, dPhiAB + dPhiBC, dPhiBC, frA, frB, clctA, dTheta, rpcA, rpcB, rpcC};
    else if (mode == 11)
      predictors = {theta, St1_ring2, dPhiBC, dPhiAB, dPhiAB + dPhiBC, frA, frB, clctA, dTheta, rpcA, rpcB, rpcC};
    else if (mode == 7)
      predictors = {theta, dPhiAB, dPhiBC, dPhiAB + dPhiBC, frA, clctA, dTheta, rpcA, rpcB, rpcC};
  } else if (nHits == 2 && mode >= 8) {
    predictors = {theta, St1_ring2, dPhiAB, frA, frB, clctA, clctB, dTheta, (clctA == 0), (clctB == 0)};
  } else if (nHits == 2 && mode < 8) {
    predictors = {theta, dPhiAB, frA, frB, clctA, clctB, dTheta, (clctA == 0), (clctB == 0)};
  } else {
    emtf_assert(false && "Incorrect nHits or mode");
  }

  // Retreive pT from XMLs
  std::vector<double> tree_data(predictors.cbegin(), predictors.cend());

  auto tree_event = std::make_unique<emtf::Event>();
  tree_event->predictedValue = 0;
  tree_event->data = tree_data;

  // forests_.at(mode).predictEvent(tree_event.get(), 400);
  emtf::Forest& forest = const_cast<emtf::Forest&>(forests_.at(mode));
  forest.predictEvent(tree_event.get(), 400);

  // // Adjust this for different XMLs
  if (ptLUTVersion_ >= 8) {  // Run 3 2022 BDT uses log2(pT) target
    float log2_pt = tree_event->predictedValue;
    pt_xml = pow(2, fmax(0.0, log2_pt));  // Protect against negative values
  } else if (ptLUTVersion_ >= 6) {        // Run 2 2017/2018 BDTs use 1/pT target
    float inv_pt = tree_event->predictedValue;
    pt_xml = 1.0 / fmax(0.001, inv_pt);  // Protect against negative values
  }

  return pt_xml;

}  // End function: float PtAssignmentEngine2017::calculate_pt_xml(const address_t& address)

// Calculate XML pT directly from track quantities, without forming an address
float PtAssignmentEngine2017::calculate_pt_xml(const EMTFTrack& track) const {
  float pt_xml = 0.;

  EMTFPtLUT data = track.PtLUT();

  auto contain = [](const std::vector<int>& vec, int elem) {
    return (std::find(vec.begin(), vec.end(), elem) != vec.end());
  };

  int endcap = track.Endcap();
  int mode = track.Mode();
  int theta = track.Theta_fp();
  int phi = track.Phi_fp();
  if (!contain(allowedModes_, mode))
    return pt_xml;

  // Which stations have hits
  int st1 = (mode >= 8);
  int st2 = ((mode % 8) >= 4);
  int st3 = ((mode % 4) >= 2);
  int st4 = ((mode % 2) == 1);

  // Variables for input to XMLs
  int dPhi_12, dPhi_13, dPhi_14, dPhi_23, dPhi_24, dPhi_34, dPhiSign;
  int dPhiSum4, dPhiSum4A, dPhiSum3, dPhiSum3A, outStPhi;
  int dTh_12, dTh_13, dTh_14, dTh_23, dTh_24, dTh_34;
  int FR_1, FR_2, FR_3, FR_4;
  int bend_1, bend_2, bend_3, bend_4;
  int RPC_1, RPC_2, RPC_3, RPC_4;
  int St1_ring2 = data.st1_ring2;

  int ph1 = -99, ph2 = -99, ph3 = -99, ph4 = -99;
  int th1 = -99, th2 = -99, th3 = -99, th4 = -99;
  int pat1 = -99, pat2 = -99, pat3 = -99, pat4 = -99;

  // Compute the original phi and theta coordinates
  if (st2) {
    ph2 = phi;    // Track phi is from station 2 (if it exists), otherwise 3 or 4
    th2 = theta;  // Likewise for track theta
    if (st1)
      ph1 = ph2 - data.delta_ph[0] * (data.sign_ph[0] ? 1 : -1);
    if (st1)
      th1 = th2 - data.delta_th[0] * (data.sign_th[0] ? 1 : -1);
    if (st3)
      ph3 = ph2 + data.delta_ph[3] * (data.sign_ph[3] ? 1 : -1);
    // Important that phi be from adjacent station pairs (see note below)
    if (st3 && st4)
      ph4 = ph3 + data.delta_ph[5] * (data.sign_ph[5] ? 1 : -1);
    else if (st4)
      ph4 = ph2 + data.delta_ph[4] * (data.sign_ph[4] ? 1 : -1);
    // Important that theta be from first-last station pair, not adjacent pairs: delta_th values are "best" for each pair, but
    // thanks to duplicated CSC LCTs, are not necessarily consistent (or physical) between pairs or between delta_th and delta_ph.
    // This is an artifact of the firmware implementation of deltas: see src/AngleCalculation.cc.
    if (st1 && st3)
      th3 = th1 + data.delta_th[1] * (data.sign_th[1] ? 1 : -1);
    else if (st3)
      th3 = th2 + data.delta_th[3] * (data.sign_th[3] ? 1 : -1);
    if (st1 && st4)
      th4 = th1 + data.delta_th[2] * (data.sign_th[2] ? 1 : -1);
    else if (st4)
      th4 = th2 + data.delta_th[4] * (data.sign_th[4] ? 1 : -1);
  } else if (st3) {
    ph3 = phi;
    th3 = theta;
    if (st1)
      ph1 = ph3 - data.delta_ph[1] * (data.sign_ph[1] ? 1 : -1);
    if (st1)
      th1 = th3 - data.delta_th[1] * (data.sign_th[1] ? 1 : -1);
    if (st4)
      ph4 = ph3 + data.delta_ph[5] * (data.sign_ph[5] ? 1 : -1);
    if (st1 && st4)
      th4 = th1 + data.delta_th[2] * (data.sign_th[2] ? 1 : -1);
    else if (st4)
      th4 = th3 + data.delta_th[5] * (data.sign_th[5] ? 1 : -1);
  } else if (st4) {
    ph4 = phi;
    th4 = theta;
    if (st1)
      ph1 = ph4 - data.delta_ph[2] * (data.sign_ph[2] ? 1 : -1);
    if (st1)
      th1 = th4 - data.delta_th[2] * (data.sign_th[2] ? 1 : -1);
  }

  if (st1)
    pat1 = data.cpattern[0];
  if (st2)
    pat2 = data.cpattern[1];
  if (st3)
    pat3 = data.cpattern[2];
  if (st4)
    pat4 = data.cpattern[3];

  // BEGIN: Identical (almost) to BDT training code in EMTFPtAssign2017/PtRegression_Apr_2017.C

  theta = aux().calcTrackTheta(th1, th2, th3, th4, St1_ring2, mode, true);

  aux().calcDeltaPhis(dPhi_12,
                      dPhi_13,
                      dPhi_14,
                      dPhi_23,
                      dPhi_24,
                      dPhi_34,
                      dPhiSign,
                      dPhiSum4,
                      dPhiSum4A,
                      dPhiSum3,
                      dPhiSum3A,
                      outStPhi,
                      ph1,
                      ph2,
                      ph3,
                      ph4,
                      mode,
                      true);

  aux().calcDeltaThetas(dTh_12, dTh_13, dTh_14, dTh_23, dTh_24, dTh_34, th1, th2, th3, th4, mode, true);

  FR_1 = (st1 ? data.fr[0] : -99);
  FR_2 = (st2 ? data.fr[1] : -99);
  FR_3 = (st3 ? data.fr[2] : -99);
  FR_4 = (st4 ? data.fr[3] : -99);

  aux().calcBends(bend_1, bend_2, bend_3, bend_4, pat1, pat2, pat3, pat4, dPhiSign, endcap, mode, true);

  RPC_1 = (st1 ? (pat1 == 0) : -99);
  RPC_2 = (st2 ? (pat2 == 0) : -99);
  RPC_3 = (st3 ? (pat3 == 0) : -99);
  RPC_4 = (st4 ? (pat4 == 0) : -99);

  aux().calcRPCs(RPC_1, RPC_2, RPC_3, RPC_4, mode, St1_ring2, theta, true);

  // END: Identical (almost) to BDT training code in EMTFPtAssign2017/PtRegression_Apr_2017.C

  // Fill vectors of variables for XMLs
  // KK: sequence of variables here should exaclty match <Variables> block produced by TMVA
  std::vector<int> predictors;
  switch (mode) {
    case 15:  // 1-2-3-4
      predictors = {theta,    St1_ring2, dPhi_12,  dPhi_23,   dPhi_34,  dPhi_13, dPhi_14, dPhi_24, FR_1,  bend_1,
                    dPhiSum4, dPhiSum4A, dPhiSum3, dPhiSum3A, outStPhi, dTh_14,  RPC_1,   RPC_2,   RPC_3, RPC_4};
      break;
    case 14:  // 1-2-3
      predictors = {theta, St1_ring2, dPhi_12, dPhi_23, dPhi_13, FR_1, FR_2, bend_1, dTh_13, RPC_1, RPC_2, RPC_3};
      break;
    case 13:  // 1-2-4
      predictors = {theta, St1_ring2, dPhi_12, dPhi_14, dPhi_24, FR_1, FR_2, bend_1, dTh_14, RPC_1, RPC_2, RPC_4};
      break;
    case 11:  // 1-3-4
      predictors = {theta, St1_ring2, dPhi_34, dPhi_13, dPhi_14, FR_1, FR_3, bend_1, dTh_14, RPC_1, RPC_3, RPC_4};
      break;
    case 7:  // 2-3-4
      predictors = {theta, dPhi_23, dPhi_34, dPhi_24, FR_2, bend_2, dTh_24, RPC_2, RPC_3, RPC_4};
      break;
    case 12:  // 1-2
      predictors = {theta, St1_ring2, dPhi_12, FR_1, FR_2, bend_1, bend_2, dTh_12, RPC_1, RPC_2};
      break;
    case 10:  // 1-3
      predictors = {theta, St1_ring2, dPhi_13, FR_1, FR_3, bend_1, bend_3, dTh_13, RPC_1, RPC_3};
      break;
    case 9:  // 1-4
      predictors = {theta, St1_ring2, dPhi_14, FR_1, FR_4, bend_1, bend_4, dTh_14, RPC_1, RPC_4};
      break;
    case 6:  // 2-3
      predictors = {theta, dPhi_23, FR_2, FR_3, bend_2, bend_3, dTh_23, RPC_2, RPC_3};
      break;
    case 5:  // 2-4
      predictors = {theta, dPhi_24, FR_2, FR_4, bend_2, bend_4, dTh_24, RPC_2, RPC_4};
      break;
    case 3:  // 3-4
      predictors = {theta, dPhi_34, FR_3, FR_4, bend_3, bend_4, dTh_34, RPC_3, RPC_4};
      break;
  }

  std::vector<double> tree_data(predictors.cbegin(), predictors.cend());

  auto tree_event = std::make_unique<emtf::Event>();
  tree_event->predictedValue = 0;
  tree_event->data = tree_data;

  // forests_.at(mode).predictEvent(tree_event.get(), 400);
  emtf::Forest& forest = const_cast<emtf::Forest&>(forests_.at(mode));
  forest.predictEvent(tree_event.get(), 400);

  // // Adjust this for different XMLs
  if (ptLUTVersion_ >= 8) {  // Run 3 2022 BDT uses log2(pT) target
    float log2_pt = tree_event->predictedValue;
    pt_xml = pow(2, fmax(0.0, log2_pt));  // Protect against negative values
  } else if (ptLUTVersion_ >= 6) {        // Run 2 2017/2018 BDTs use 1/pT target
    float inv_pt = tree_event->predictedValue;
    pt_xml = 1.0 / fmax(0.001, inv_pt);  // Protect against negative values
  }

  return pt_xml;

}  // End function: float PtAssignmentEngine2017::calculate_pt_xml(const EMTFTrack& track)
