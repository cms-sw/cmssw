// Class Header
#include "MuonSeeddPhiScale.h"

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TVector3.h"

#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <string>
#include <stdio.h>
#include <algorithm>

using namespace std;
using namespace edm;

// constructors
MuonSeeddPhiScale::MuonSeeddPhiScale(const ParameterSet& pset) {
  // dPhi scale factors
  CSC01_1 = pset.getParameter<std::vector<double> >("CSC_01_1_scale");
  CSC12_1 = pset.getParameter<std::vector<double> >("CSC_12_1_scale");
  CSC12_2 = pset.getParameter<std::vector<double> >("CSC_12_2_scale");
  CSC12_3 = pset.getParameter<std::vector<double> >("CSC_12_3_scale");
  CSC13_2 = pset.getParameter<std::vector<double> >("CSC_13_2_scale");
  CSC13_3 = pset.getParameter<std::vector<double> >("CSC_13_3_scale");
  CSC14_3 = pset.getParameter<std::vector<double> >("CSC_14_3_scale");
  CSC23_1 = pset.getParameter<std::vector<double> >("CSC_23_1_scale");
  CSC23_2 = pset.getParameter<std::vector<double> >("CSC_23_2_scale");
  CSC24_1 = pset.getParameter<std::vector<double> >("CSC_24_1_scale");
  CSC34_1 = pset.getParameter<std::vector<double> >("CSC_34_1_scale");

  DT12_1 = pset.getParameter<std::vector<double> >("DT_12_1_scale");
  DT12_2 = pset.getParameter<std::vector<double> >("DT_12_2_scale");
  DT13_1 = pset.getParameter<std::vector<double> >("DT_13_1_scale");
  DT13_2 = pset.getParameter<std::vector<double> >("DT_13_2_scale");
  DT14_1 = pset.getParameter<std::vector<double> >("DT_14_1_scale");
  DT14_2 = pset.getParameter<std::vector<double> >("DT_14_2_scale");
  DT23_1 = pset.getParameter<std::vector<double> >("DT_23_1_scale");
  DT23_2 = pset.getParameter<std::vector<double> >("DT_23_2_scale");
  DT24_1 = pset.getParameter<std::vector<double> >("DT_24_1_scale");
  DT24_2 = pset.getParameter<std::vector<double> >("DT_24_2_scale");
  DT34_1 = pset.getParameter<std::vector<double> >("DT_34_1_scale");
  DT34_2 = pset.getParameter<std::vector<double> >("DT_34_2_scale");

  OL1213 = pset.getParameter<std::vector<double> >("OL_1213_0_scale");
  OL1222 = pset.getParameter<std::vector<double> >("OL_1222_0_scale");
  OL1232 = pset.getParameter<std::vector<double> >("OL_1232_0_scale");
  OL2213 = pset.getParameter<std::vector<double> >("OL_2213_0_scale");
  OL2222 = pset.getParameter<std::vector<double> >("OL_2222_0_scale");

  SMB_10S = pset.getParameter<std::vector<double> >("SMB_10_0_scale");
  SMB_11S = pset.getParameter<std::vector<double> >("SMB_11_0_scale");
  SMB_12S = pset.getParameter<std::vector<double> >("SMB_12_0_scale");
  SMB_20S = pset.getParameter<std::vector<double> >("SMB_20_0_scale");
  SMB_21S = pset.getParameter<std::vector<double> >("SMB_21_0_scale");
  SMB_22S = pset.getParameter<std::vector<double> >("SMB_22_0_scale");
  SMB_30S = pset.getParameter<std::vector<double> >("SMB_30_0_scale");
  SMB_31S = pset.getParameter<std::vector<double> >("SMB_31_0_scale");
  SMB_32S = pset.getParameter<std::vector<double> >("SMB_32_0_scale");

  SME_11S = pset.getParameter<std::vector<double> >("SME_11_0_scale");
  SME_12S = pset.getParameter<std::vector<double> >("SME_12_0_scale");
  SME_13S = pset.getParameter<std::vector<double> >("SME_13_0_scale");
  SME_21S = pset.getParameter<std::vector<double> >("SME_21_0_scale");
  SME_22S = pset.getParameter<std::vector<double> >("SME_22_0_scale");
}

// destructor
MuonSeeddPhiScale::~MuonSeeddPhiScale() {}

void MuonSeeddPhiScale::ScaleCSCdPhi(double dPhiP1[2][5][5], double EtaP1[2][5]) {
  // fill the information for CSC pT parameterization from segment pair
  if (dPhiP1[1][0][1] != 99.0) {
    double oPh = 1. / dPhiP1[0][0][1];
    double oPhi = 1. / dPhiP1[1][0][1];
    dPhiP1[0][0][1] = dPhiP1[0][0][1] / (1. + (CSC01_1[3] / fabs(oPh + 10.)));
    dPhiP1[1][0][1] = dPhiP1[1][0][1] / (1. + (CSC01_1[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP1[1][0][2] != 99.0 && fabs(EtaP1[1][0]) > 1.6) {
    double oPh = 1. / dPhiP1[0][0][2];
    double oPhi = 1. / dPhiP1[1][0][2];
    dPhiP1[0][0][2] = dPhiP1[0][0][2] / (1. + (CSC12_3[3] / fabs(oPh + 10.)));
    dPhiP1[1][0][2] = dPhiP1[1][0][2] / (1. + (CSC12_3[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][1][2] != 99.0 && fabs(EtaP1[1][1]) <= 1.6 && fabs(EtaP1[1][0]) > 1.2) {
    double oPh = 1. / dPhiP1[0][1][2];
    double oPhi = 1. / dPhiP1[1][1][2];
    dPhiP1[0][1][2] = dPhiP1[0][1][2] / (1. + (CSC12_2[3] / fabs(oPh + 10.)));
    dPhiP1[1][1][2] = dPhiP1[1][1][2] / (1. + (CSC12_2[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][1][2] != 99.0 && fabs(EtaP1[1][1]) <= 1.2) {
    double oPh = 1. / dPhiP1[0][1][2];
    double oPhi = 1. / dPhiP1[1][1][2];
    dPhiP1[0][1][2] = dPhiP1[0][1][2] / (1. + (CSC12_1[3] / fabs(oPh + 10.)));
    dPhiP1[1][1][2] = dPhiP1[1][1][2] / (1. + (CSC12_1[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP1[1][0][3] != 99.0) {
    double oPh = 1. / dPhiP1[0][0][3];
    double oPhi = 1. / dPhiP1[1][0][3];
    dPhiP1[0][0][3] = dPhiP1[0][0][3] / (1. + (CSC13_3[3] / fabs(oPh + 10.)));
    dPhiP1[1][0][3] = dPhiP1[1][0][3] / (1. + (CSC13_3[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][1][3] != 99.0) {
    double oPh = 1. / dPhiP1[0][1][3];
    double oPhi = 1. / dPhiP1[1][1][3];
    dPhiP1[0][1][3] = dPhiP1[0][1][3] / (1. + (CSC13_2[3] / fabs(oPh + 10.)));
    dPhiP1[1][1][3] = dPhiP1[1][1][3] / (1. + (CSC13_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP1[1][0][4] != 99.0) {
    double oPh = 1. / dPhiP1[0][0][4];
    double oPhi = 1. / dPhiP1[1][0][4];
    dPhiP1[0][0][4] = dPhiP1[0][0][4] / (1. + (CSC14_3[3] / fabs(oPh + 10.)));
    dPhiP1[1][0][4] = dPhiP1[1][0][4] / (1. + (CSC14_3[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][1][4] != 99.0) {
    double oPh = 1. / dPhiP1[0][1][4];
    double oPhi = 1. / dPhiP1[1][1][4];
    dPhiP1[0][1][4] = dPhiP1[0][1][4] / (1. + (CSC14_3[3] / fabs(oPh + 10.)));
    dPhiP1[1][1][4] = dPhiP1[1][1][4] / (1. + (CSC14_3[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP1[1][2][3] != 99.0 && fabs(EtaP1[1][2]) > 1.7) {
    double oPh = 1. / dPhiP1[0][2][3];
    double oPhi = 1. / dPhiP1[1][2][3];
    dPhiP1[0][2][3] = dPhiP1[0][2][3] / (1. + (CSC23_2[3] / fabs(oPh + 10.)));
    dPhiP1[1][2][3] = dPhiP1[1][2][3] / (1. + (CSC23_2[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][2][3] != 99.0 && fabs(EtaP1[1][2]) <= 1.7) {
    double oPh = 1. / dPhiP1[0][2][3];
    double oPhi = 1. / dPhiP1[1][2][3];
    dPhiP1[0][2][3] = dPhiP1[0][2][3] / (1. + (CSC23_1[3] / fabs(oPh + 10.)));
    dPhiP1[1][2][3] = dPhiP1[1][2][3] / (1. + (CSC23_1[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP1[1][2][4] != 99.0) {
    double oPh = 1. / dPhiP1[0][2][4];
    double oPhi = 1. / dPhiP1[1][2][4];
    dPhiP1[0][2][4] = dPhiP1[0][2][4] / (1. + (CSC24_1[3] / fabs(oPh + 10.)));
    dPhiP1[1][2][4] = dPhiP1[1][2][4] / (1. + (CSC24_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP1[1][3][4] != 99.0) {
    double oPh = 1. / dPhiP1[0][3][4];
    double oPhi = 1. / dPhiP1[1][3][4];
    dPhiP1[0][3][4] = dPhiP1[0][3][4] / (1. + (CSC34_1[3] / fabs(oPh + 10.)));
    dPhiP1[1][3][4] = dPhiP1[1][3][4] / (1. + (CSC34_1[3] / fabs(oPhi + 10.)));
  }
}

void MuonSeeddPhiScale::ScaleDTdPhi(double dPhiP3[2][5][5], double EtaP3[2][5]) {
  /// For DT
  //  fill the information for DT pT parameterization from segment pair
  if (dPhiP3[1][1][2] != 99.0 && fabs(EtaP3[1][1]) <= 0.7) {
    double oPh = 1. / dPhiP3[0][1][2];
    double oPhi = 1. / dPhiP3[1][1][2];
    dPhiP3[0][1][2] = dPhiP3[0][1][2] / (1. + (DT12_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][2] = dPhiP3[1][1][2] / (1. + (DT12_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][1][2] != 99.0 && fabs(EtaP3[1][1]) > 0.7) {
    double oPh = 1. / dPhiP3[0][1][2];
    double oPhi = 1. / dPhiP3[1][1][2];
    dPhiP3[0][1][2] = dPhiP3[0][1][2] / (1. + (DT12_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][2] = dPhiP3[1][1][2] / (1. + (DT12_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP3[1][1][3] != 99.0 && fabs(EtaP3[1][1]) <= 0.6) {
    double oPh = 1. / dPhiP3[0][1][3];
    double oPhi = 1. / dPhiP3[1][1][3];
    dPhiP3[0][1][3] = dPhiP3[0][1][3] / (1. + (DT13_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][3] = dPhiP3[1][1][3] / (1. + (DT13_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][1][3] != 99.0 && fabs(EtaP3[1][1]) > 0.6) {
    double oPh = 1. / dPhiP3[0][1][3];
    double oPhi = 1. / dPhiP3[1][1][3];
    dPhiP3[0][1][3] = dPhiP3[0][1][3] / (1. + (DT13_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][3] = dPhiP3[1][1][3] / (1. + (DT13_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP3[1][1][4] != 99.0 && fabs(EtaP3[1][1]) <= 0.52) {
    double oPh = 1. / dPhiP3[0][1][4];
    double oPhi = 1. / dPhiP3[1][1][4];
    dPhiP3[0][1][4] = dPhiP3[0][1][4] / (1. + (DT14_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][4] = dPhiP3[1][1][4] / (1. + (DT14_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][1][4] != 99.0 && fabs(EtaP3[1][1]) > 0.52) {
    double oPh = 1. / dPhiP3[0][1][4];
    double oPhi = 1. / dPhiP3[1][1][4];
    dPhiP3[0][1][4] = dPhiP3[0][1][4] / (1. + (DT14_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][1][4] = dPhiP3[1][1][4] / (1. + (DT14_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP3[1][2][3] != 99.0 && fabs(EtaP3[1][2]) <= 0.6) {
    double oPh = 1. / dPhiP3[0][2][3];
    double oPhi = 1. / dPhiP3[1][2][3];
    dPhiP3[0][2][3] = dPhiP3[0][2][3] / (1. + (DT23_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][2][3] = dPhiP3[1][2][3] / (1. + (DT23_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][2][3] != 99.0 && fabs(EtaP3[1][2]) > 0.6) {
    double oPh = 1. / dPhiP3[0][2][3];
    double oPhi = 1. / dPhiP3[1][2][3];
    dPhiP3[0][2][3] = dPhiP3[0][2][3] / (1. + (DT23_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][2][3] = dPhiP3[1][2][3] / (1. + (DT23_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP3[1][2][4] != 99.0 && fabs(EtaP3[1][2]) <= 0.52) {
    double oPh = 1. / dPhiP3[0][2][4];
    double oPhi = 1. / dPhiP3[1][2][4];
    dPhiP3[0][2][4] = dPhiP3[0][2][4] / (1. + (DT24_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][2][4] = dPhiP3[1][2][4] / (1. + (DT24_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][2][4] != 99.0 && fabs(EtaP3[1][2]) > 0.52) {
    double oPh = 1. / dPhiP3[0][2][4];
    double oPhi = 1. / dPhiP3[1][2][4];
    dPhiP3[0][2][4] = dPhiP3[0][2][4] / (1. + (DT24_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][2][4] = dPhiP3[1][2][4] / (1. + (DT24_2[3] / fabs(oPhi + 10.)));
  }

  if (dPhiP3[1][3][4] != 99.0 && fabs(EtaP3[1][3]) <= 0.51) {
    double oPh = 1. / dPhiP3[0][3][4];
    double oPhi = 1. / dPhiP3[1][3][4];
    dPhiP3[0][3][4] = dPhiP3[0][3][4] / (1. + (DT34_1[3] / fabs(oPh + 10.)));
    dPhiP3[1][3][4] = dPhiP3[1][3][4] / (1. + (DT34_1[3] / fabs(oPhi + 10.)));
  }
  if (dPhiP3[1][3][4] != 99.0 && fabs(EtaP3[1][3]) > 0.51) {
    double oPh = 1. / dPhiP3[0][3][4];
    double oPhi = 1. / dPhiP3[1][3][4];
    dPhiP3[0][3][4] = dPhiP3[0][3][4] / (1. + (DT34_2[3] / fabs(oPh + 10.)));
    dPhiP3[1][3][4] = dPhiP3[1][3][4] / (1. + (DT34_2[3] / fabs(oPhi + 10.)));
  }
}

void MuonSeeddPhiScale::ScaleOLdPhi(double dPhiP2[2][5][5], bool MBPath[2][5][3], bool MEPath[2][5][4]) {
  if (MBPath[1][1][2] && MEPath[1][1][3]) {
    double oPh = 1. / dPhiP2[0][1][1];
    double oPhi = 1. / dPhiP2[1][1][1];
    dPhiP2[0][1][1] = dPhiP2[0][1][1] / (1. + (OL1213[3] / fabs(oPh + 10.)));
    dPhiP2[1][1][1] = dPhiP2[1][1][1] / (1. + (OL1213[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][1][2] && MEPath[1][2][2]) {
    double oPh = 1. / dPhiP2[0][1][2];
    double oPhi = 1. / dPhiP2[1][1][2];
    dPhiP2[0][1][2] = dPhiP2[0][1][2] / (1. + (OL1222[3] / fabs(oPh + 10.)));
    dPhiP2[1][1][2] = dPhiP2[1][1][2] / (1. + (OL1222[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][1][2] && MEPath[1][3][2]) {
    double oPh = 1. / dPhiP2[0][1][3];
    double oPhi = 1. / dPhiP2[1][1][3];
    dPhiP2[0][1][3] = dPhiP2[0][1][3] / (1. + (OL1232[3] / fabs(oPh + 10.)));
    dPhiP2[1][1][3] = dPhiP2[1][1][3] / (1. + (OL1232[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][2][2] && MEPath[1][1][3]) {
    double oPh = 1. / dPhiP2[0][2][1];
    double oPhi = 1. / dPhiP2[1][2][1];
    dPhiP2[0][2][1] = dPhiP2[0][2][1] / (1. + (OL2213[3] / fabs(oPh + 10.)));
    dPhiP2[1][2][1] = dPhiP2[1][2][1] / (1. + (OL2213[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][2][2] && MEPath[1][2][2]) {
    double oPh = 1. / dPhiP2[0][2][2];
    double oPhi = 1. / dPhiP2[1][2][2];
    dPhiP2[0][2][2] = dPhiP2[0][2][2] / (1. + (OL2222[3] / fabs(oPh + 10.)));
    dPhiP2[1][2][2] = dPhiP2[1][2][1] / (1. + (OL2222[3] / fabs(oPhi + 10.)));
  }
}

void MuonSeeddPhiScale::ScaleMESingle(double ME_phi[2][5][4], bool MEPath[2][5][4]) {
  if (MEPath[1][0][1] && MEPath[0][0][1]) {
    double oPh = 1. / ME_phi[0][0][1];
    double oPhi = 1. / ME_phi[1][0][1];
    ME_phi[0][0][1] = ME_phi[0][0][1] / (1. + (SME_11S[3] / fabs(oPh + 10.)));
    ME_phi[1][0][1] = ME_phi[1][0][1] / (1. + (SME_11S[3] / fabs(oPhi + 10.)));
  }
  if (MEPath[1][1][2] && MEPath[0][1][2]) {
    double oPh = 1. / ME_phi[0][1][2];
    double oPhi = 1. / ME_phi[1][1][2];
    ME_phi[0][1][2] = ME_phi[0][1][2] / (1. + (SME_12S[3] / fabs(oPh + 10.)));
    ME_phi[1][1][2] = ME_phi[1][1][2] / (1. + (SME_12S[3] / fabs(oPhi + 10.)));
  }
  if (MEPath[1][1][3] && MEPath[0][1][3]) {
    double oPh = 1. / ME_phi[0][1][3];
    double oPhi = 1. / ME_phi[1][1][3];
    ME_phi[0][1][3] = ME_phi[0][1][3] / (1. + (SME_13S[3] / fabs(oPh + 10.)));
    ME_phi[1][1][3] = ME_phi[1][1][3] / (1. + (SME_13S[3] / fabs(oPhi + 10.)));
  }
  if (MEPath[1][2][1] && MEPath[0][2][1]) {
    double oPh = 1. / ME_phi[0][2][1];
    double oPhi = 1. / ME_phi[1][2][1];
    ME_phi[0][2][1] = ME_phi[0][2][1] / (1. + (SME_21S[3] / fabs(oPh + 10.)));
    ME_phi[1][2][1] = ME_phi[1][2][1] / (1. + (SME_21S[3] / fabs(oPhi + 10.)));
  }
  if (MEPath[1][2][2] && MEPath[0][2][2]) {
    double oPh = 1. / ME_phi[0][2][2];
    double oPhi = 1. / ME_phi[1][2][2];
    ME_phi[0][2][2] = ME_phi[0][2][2] / (1. + (SME_22S[3] / fabs(oPh + 10.)));
    ME_phi[1][2][2] = ME_phi[1][2][2] / (1. + (SME_22S[3] / fabs(oPhi + 10.)));
  }
}

void MuonSeeddPhiScale::ScaleMBSingle(double MB_phi[2][5][3], bool MBPath[2][5][3]) {
  if (MBPath[1][1][0] && MBPath[0][1][0]) {
    double oPh = 1. / MB_phi[0][1][0];
    double oPhi = 1. / MB_phi[1][1][0];
    MB_phi[0][1][0] = MB_phi[0][1][0] / (1. + (SMB_10S[3] / fabs(oPh + 10.)));
    MB_phi[1][1][0] = MB_phi[1][1][0] / (1. + (SMB_10S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][1][1] && MBPath[0][1][1]) {
    double oPh = 1. / MB_phi[0][1][1];
    double oPhi = 1. / MB_phi[1][1][1];
    MB_phi[0][1][1] = MB_phi[0][1][1] / (1. + (SMB_11S[3] / fabs(oPh + 10.)));
    MB_phi[1][1][1] = MB_phi[1][1][1] / (1. + (SMB_11S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][1][2] && MBPath[0][1][2]) {
    double oPh = 1. / MB_phi[0][1][2];
    double oPhi = 1. / MB_phi[1][1][2];
    MB_phi[0][1][2] = MB_phi[0][1][2] / (1. + (SMB_12S[3] / fabs(oPh + 10.)));
    MB_phi[1][1][2] = MB_phi[1][1][2] / (1. + (SMB_12S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][2][0] && MBPath[0][2][0]) {
    double oPh = 1. / MB_phi[0][2][0];
    double oPhi = 1. / MB_phi[1][2][0];
    MB_phi[0][2][0] = MB_phi[0][2][0] / (1. + (SMB_20S[3] / fabs(oPh + 10.)));
    MB_phi[1][2][0] = MB_phi[1][2][0] / (1. + (SMB_20S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][2][1] && MBPath[0][2][1]) {
    double oPh = 1. / MB_phi[0][2][1];
    double oPhi = 1. / MB_phi[1][2][1];
    MB_phi[0][2][1] = MB_phi[0][2][1] / (1. + (SMB_21S[3] / fabs(oPh + 10.)));
    MB_phi[1][2][1] = MB_phi[1][2][1] / (1. + (SMB_21S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][2][2] && MBPath[0][2][2]) {
    double oPh = 1. / MB_phi[0][2][2];
    double oPhi = 1. / MB_phi[1][2][2];
    MB_phi[0][2][2] = MB_phi[0][2][2] / (1. + (SMB_22S[3] / fabs(oPh + 10.)));
    MB_phi[1][2][2] = MB_phi[1][2][2] / (1. + (SMB_22S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][3][0] && MBPath[0][3][0]) {
    double oPh = 1. / MB_phi[0][3][0];
    double oPhi = 1. / MB_phi[1][3][0];
    MB_phi[0][3][0] = MB_phi[0][3][0] / (1. + (SMB_30S[3] / fabs(oPh + 10.)));
    MB_phi[1][3][0] = MB_phi[1][3][0] / (1. + (SMB_30S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][3][1] && MBPath[0][3][1]) {
    double oPh = 1. / MB_phi[0][3][1];
    double oPhi = 1. / MB_phi[1][3][1];
    MB_phi[0][3][1] = MB_phi[0][3][1] / (1. + (SMB_31S[3] / fabs(oPh + 10.)));
    MB_phi[1][3][1] = MB_phi[1][3][1] / (1. + (SMB_31S[3] / fabs(oPhi + 10.)));
  }
  if (MBPath[1][3][2] && MBPath[0][3][2]) {
    double oPh = 1. / MB_phi[0][3][2];
    double oPhi = 1. / MB_phi[1][3][2];
    MB_phi[0][3][2] = MB_phi[0][3][2] / (1. + (SMB_32S[3] / fabs(oPh + 10.)));
    MB_phi[1][3][2] = MB_phi[1][3][2] / (1. + (SMB_32S[3] / fabs(oPhi + 10.)));
  }
}
