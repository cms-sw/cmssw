// Class Header
#include "MuonSeedParaFillHisto.h"

#include "TFile.h"
#include "TVector3.h"

#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <string>
#include <stdio.h>
#include <algorithm>

//DEFINE_FWK_MODULE(MuonSeedParaFillHisto);
using namespace std;
using namespace edm;

// constructors
MuonSeedParaFillHisto::MuonSeedParaFillHisto() {}

// destructor
MuonSeedParaFillHisto::~MuonSeedParaFillHisto() {}

void MuonSeedParaFillHisto::FillCSCSegmentPair(
    H2DRecHit2* histo2, double pt1[5], double chi2_dof1[5], double dPhiP1[2][5][5], double EtaP1[2][5]) {
  double ptIn = (pt1[1] != 0.) ? pt1[1] : pt1[2];
  if (ptIn == 0)
    ptIn = pt1[0];

  // fill the information for CSC pT parameterization from segment pair
  if (dPhiP1[1][0][1] != 99.0 && chi2_dof1[0] < 2000.0 && chi2_dof1[1] < 2000.0) {
    histo2->Fill5_0(dPhiP1[0][0][1],
                    dPhiP1[1][0][1],
                    ptIn * dPhiP1[0][0][1],
                    ptIn * dPhiP1[1][0][1],
                    fabs(EtaP1[0][0]),
                    fabs(EtaP1[1][0]));
  }
  if (dPhiP1[1][0][2] != 99.0 && chi2_dof1[0] < 2000.0 && chi2_dof1[2] < 2000.0) {
    histo2->Fill5_1(dPhiP1[0][0][2],
                    dPhiP1[1][0][2],
                    ptIn * dPhiP1[0][0][2],
                    ptIn * dPhiP1[1][0][2],
                    fabs(EtaP1[0][0]),
                    fabs(EtaP1[1][0]));
  }
  if (dPhiP1[1][0][3] != 99.0 && chi2_dof1[0] < 2000.0 && chi2_dof1[3] < 2000.0) {
    histo2->Fill5_2(dPhiP1[0][0][3],
                    dPhiP1[1][0][3],
                    ptIn * dPhiP1[0][0][3],
                    ptIn * dPhiP1[1][0][3],
                    fabs(EtaP1[0][0]),
                    fabs(EtaP1[1][0]));
  }
  if (dPhiP1[1][0][4] != 99.0 && chi2_dof1[0] < 2000.0 && chi2_dof1[4] < 2000.0) {
    histo2->Fill5_3(dPhiP1[0][0][4],
                    dPhiP1[1][0][4],
                    ptIn * dPhiP1[0][0][4],
                    ptIn * dPhiP1[1][0][4],
                    fabs(EtaP1[0][0]),
                    fabs(EtaP1[1][0]));
  }
  if (dPhiP1[1][1][2] != 99.0 && chi2_dof1[1] < 2000.0 && chi2_dof1[2] < 2000.0) {
    histo2->Fill5_1(dPhiP1[0][1][2],
                    dPhiP1[1][1][2],
                    ptIn * dPhiP1[0][1][2],
                    ptIn * dPhiP1[1][1][2],
                    fabs(EtaP1[0][1]),
                    fabs(EtaP1[1][1]));
  }
  if (dPhiP1[1][1][3] != 99.0 && chi2_dof1[1] < 2000.0 && chi2_dof1[3] < 2000.0) {
    histo2->Fill5_2(dPhiP1[0][1][3],
                    dPhiP1[1][1][3],
                    ptIn * dPhiP1[0][1][3],
                    ptIn * dPhiP1[1][1][3],
                    fabs(EtaP1[0][1]),
                    fabs(EtaP1[1][1]));
  }
  if (dPhiP1[1][1][4] != 99.0 && chi2_dof1[1] < 2000.0 && chi2_dof1[4] < 2000.0) {
    histo2->Fill5_3(dPhiP1[0][1][4],
                    dPhiP1[1][1][4],
                    ptIn * dPhiP1[0][1][4],
                    ptIn * dPhiP1[1][1][4],
                    fabs(EtaP1[0][1]),
                    fabs(EtaP1[1][1]));
  }
  if (dPhiP1[1][2][3] != 99.0 && chi2_dof1[2] < 2000.0 && chi2_dof1[3] < 2000.0) {
    histo2->Fill5_4(dPhiP1[0][2][3],
                    dPhiP1[1][2][3],
                    ptIn * dPhiP1[0][2][3],
                    ptIn * dPhiP1[1][2][3],
                    fabs(EtaP1[0][2]),
                    fabs(EtaP1[1][2]));
  }
  if (dPhiP1[1][2][4] != 99.0 && chi2_dof1[2] < 2000.0 && chi2_dof1[4] < 2000.0) {
    histo2->Fill5_5(dPhiP1[0][2][4],
                    dPhiP1[1][2][4],
                    ptIn * dPhiP1[0][2][4],
                    ptIn * dPhiP1[1][2][4],
                    fabs(EtaP1[0][2]),
                    fabs(EtaP1[1][2]));
  }
  if (dPhiP1[1][3][4] != 99.0 && chi2_dof1[3] < 2000.0 && chi2_dof1[4] < 2000.0) {
    histo2->Fill5_6(dPhiP1[0][3][4],
                    dPhiP1[1][3][4],
                    ptIn * dPhiP1[0][3][4],
                    ptIn * dPhiP1[1][3][4],
                    fabs(EtaP1[0][3]),
                    fabs(EtaP1[1][3]));
  }
}

void MuonSeedParaFillHisto::FillDTSegmentPair(
    H2DRecHit3* histo3, double pt1[5], double chi2_dof3[5], double dPhiP3[2][5][5], double EtaP3[2][5]) {
  double ptIn = (pt1[1] != 0.) ? pt1[1] : pt1[2];
  if (ptIn == 0)
    ptIn = pt1[0];

  /// For DT
  //  fill the information for DT pT parameterization from segment pair
  if (dPhiP3[1][1][2] != 99.0 && chi2_dof3[1] < 2000.0 && chi2_dof3[2] < 2000.0) {
    histo3->Fill6_1(dPhiP3[0][1][2],
                    dPhiP3[1][1][2],
                    ptIn * dPhiP3[0][1][2],
                    ptIn * dPhiP3[1][1][2],
                    fabs(EtaP3[0][1]),
                    fabs(EtaP3[1][1]));
  }
  if (dPhiP3[1][1][3] != 99.0 && chi2_dof3[1] < 2000.0 && chi2_dof3[3] < 2000.0) {
    histo3->Fill6_2(dPhiP3[0][1][3],
                    dPhiP3[1][1][3],
                    ptIn * dPhiP3[0][1][3],
                    ptIn * dPhiP3[1][1][3],
                    fabs(EtaP3[0][1]),
                    fabs(EtaP3[1][1]));
  }
  if (dPhiP3[1][1][4] != 99.0 && chi2_dof3[1] < 2000.0 && chi2_dof3[4] < 2000.0) {
    histo3->Fill6_3(dPhiP3[0][1][4],
                    dPhiP3[1][1][4],
                    ptIn * dPhiP3[0][1][4],
                    ptIn * dPhiP3[1][1][4],
                    fabs(EtaP3[0][1]),
                    fabs(EtaP3[1][1]));
  }
  if (dPhiP3[1][2][3] != 99.0 && chi2_dof3[2] < 2000.0 && chi2_dof3[3] < 2000.0) {
    histo3->Fill6_4(dPhiP3[0][2][3],
                    dPhiP3[1][2][3],
                    ptIn * dPhiP3[0][2][3],
                    ptIn * dPhiP3[1][2][3],
                    fabs(EtaP3[0][2]),
                    fabs(EtaP3[1][2]));
  }
  if (dPhiP3[1][2][4] != 99.0 && chi2_dof3[2] < 2000.0 && chi2_dof3[4] < 2000.0) {
    histo3->Fill6_5(dPhiP3[0][2][4],
                    dPhiP3[1][2][4],
                    ptIn * dPhiP3[0][2][4],
                    ptIn * dPhiP3[1][2][4],
                    fabs(EtaP3[0][2]),
                    fabs(EtaP3[1][2]));
  }
  if (dPhiP3[1][3][4] != 99.0 && chi2_dof3[3] < 2000.0 && chi2_dof3[4] < 2000.0) {
    histo3->Fill6_6(dPhiP3[0][3][4],
                    dPhiP3[1][3][4],
                    ptIn * dPhiP3[0][3][4],
                    ptIn * dPhiP3[1][3][4],
                    fabs(EtaP3[0][3]),
                    fabs(EtaP3[1][3]));
  }
}

void MuonSeedParaFillHisto::FillCSCSegmentPairByChamber(H2DRecHit4* hME1[15],
                                                        double pt1[5],
                                                        double dPhiP1[2][5][5],
                                                        double EtaP1[2][5],
                                                        bool MEPath[2][5][4],
                                                        double dEtaP1[2][5][5]) {
  H2DRecHit4* histo4 = 0;
  //  Look at different Bxdl btw. stations & rings
  /// All possible segment pairs in CSC
  ///                 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  int csc1[2][15] = {{1, 1, 12, 12, 13, 1, 1, 12, 13, 1, 21, 21, 22, 21, 31},
                     {12, 21, 21, 22, 22, 31, 32, 32, 32, 41, 31, 32, 32, 41, 41}};

  double ptIn = (pt1[1] != 0.) ? pt1[1] : pt1[2];
  if (ptIn == 0)
    ptIn = pt1[0];

  for (int l = 0; l < 15; l++) {
    int s1 = csc1[0][l] / 10;  // 0 0 1 1 1 0 0 1 1 0 2 2 2 2 3
    int r1 = csc1[0][l] % 10;  // 1 1 2 2 3 1 1 2 3 1 1 1 2 1 1
    int s2 = csc1[1][l] / 10;
    int r2 = csc1[1][l] % 10;
    if (MEPath[1][s1][r1] && MEPath[1][s2][r2] && MEPath[0][s1][r1] && MEPath[0][s2][r2] && ptIn > 0.) {
      double ME_Resol = dPhiP1[0][s1][s2] - dPhiP1[1][s1][s2];
      histo4 = hME1[l];
      histo4->Fill8((ptIn * dPhiP1[0][s1][s2]), dPhiP1[0][s1][s2], dEtaP1[0][s1][s2], fabs(EtaP1[0][s2]), ptIn);
      histo4->Fill8a(
          (ptIn * dPhiP1[1][s1][s2]), dPhiP1[1][s1][s2], dEtaP1[1][s1][s2], fabs(EtaP1[1][s2]), ptIn, ME_Resol);
    }
  }
}

void MuonSeedParaFillHisto::FillDTSegmentPairByChamber(H2DRecHit5* hMB1[26],
                                                       double pt1[5],
                                                       double dPhiP3[2][5][5],
                                                       double EtaP3[2][5],
                                                       bool MBPath[2][5][3],
                                                       double dEtaP3[2][5][5]) {
  H2DRecHit5* histo5 = 0;
  //  Look at different Bxdl btw. stations & rings
  /// All possible segment pair in DT
  ///               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
  int dt1[2][26] = {
      {10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 20, 20, 20, 21, 21, 21, 21, 22, 22, 30, 31, 31, 32},
      {20, 30, 31, 40, 41, 21, 22, 31, 32, 41, 42, 22, 32, 30, 40, 41, 31, 32, 41, 42, 32, 42, 40, 41, 42, 42}};
  for (int l = 0; l < 26; l++) {
    int s1 = dt1[0][l] / 10;
    int w1 = dt1[0][l] % 10;
    int s2 = dt1[1][l] / 10;
    int w2 = dt1[1][l] % 10;

    double ptIn = (pt1[1] != 0.) ? pt1[1] : pt1[2];
    if (ptIn == 0)
      ptIn = pt1[0];

    if (MBPath[1][s1][w1] && MBPath[1][s2][w2] && MBPath[0][s1][w1] && MBPath[0][s2][w2] && ptIn > 0.) {
      double MB_Resol = dPhiP3[0][s1][s2] - dPhiP3[1][s1][s2];
      if (s2 != 4) {
        histo5 = hMB1[l];
        histo5->Fill9((ptIn * dPhiP3[0][s1][s2]), dPhiP3[0][s1][s2], dEtaP3[0][s1][s2], fabs(EtaP3[0][s2]), ptIn);
        histo5->Fill9a(
            (ptIn * dPhiP3[1][s1][s2]), dPhiP3[1][s1][s2], dEtaP3[1][s1][s2], fabs(EtaP3[1][s2]), ptIn, MB_Resol);
      }
      if (s2 == 4) {
        histo5 = hMB1[l];
        histo5->Fill9((ptIn * dPhiP3[0][s1][s2]), dPhiP3[0][s1][s2], dEtaP3[0][s1][s2], fabs(EtaP3[0][s1]), ptIn);
        histo5->Fill9a(
            (ptIn * dPhiP3[1][s1][s2]), dPhiP3[1][s1][s2], dEtaP3[1][s1][s2], fabs(EtaP3[1][s1]), ptIn, MB_Resol);
      }
    }
  }
}

void MuonSeedParaFillHisto::FillOLSegmentPairByChamber(H2DRecHit10* hOL1[6],
                                                       double pt1[5],
                                                       double dPhiP2[2][5][5],
                                                       double EtaP3[2][5],
                                                       bool MBPath[2][5][3],
                                                       bool MEPath[2][5][4],
                                                       double dEtaP2[2][5][5]) {
  //H2DRecHit10 *histo10;
  //  Look at different Bxdl in overlap region
  /// All possible segment pairs in overlap region
  ///              0  1  2  3  4  5
  int olp[2][6] = {{12, 12, 12, 22, 22, 32}, {13, 22, 32, 13, 22, 13}};
  for (int l = 0; l < 6; l++) {
    int s1 = olp[0][l] / 10;
    int w1 = olp[0][l] % 10;
    int s2 = olp[1][l] / 10;
    int w2 = olp[1][l] % 10;
    if (MBPath[1][s1][w1] && MEPath[1][s2][w2]) {
      double OL_Resol = dPhiP2[s1][s2] - dPhiP2[s1][s2];
      //histo10 = hOL1[l];
      hOL1[l]->Fill12((pt1[1] * dPhiP2[0][s1][s2]), dPhiP2[0][s1][s2], dEtaP2[0][s1][s2], fabs(EtaP3[0][s1]), pt1[1]);
      hOL1[l]->Fill12a(
          (pt1[1] * dPhiP2[1][s1][s2]), dPhiP2[1][s1][s2], dEtaP2[1][s1][s2], fabs(EtaP3[1][s1]), pt1[1], OL_Resol);
    }
  }
}

void MuonSeedParaFillHisto::FillCSCSegmentSingle(
    H2DRecHit6* hME2[8], double pt1[5], double ME_phi[2][5][4], double ME_eta[2][5][4], bool MEPath[2][5][4]) {
  // Fill  the 1 segment case for CSC and DT
  int csc2[8] = {1, 12, 13, 21, 22, 31, 32, 41};
  for (int l = 0; l < 8; l++) {
    int s1 = csc2[l] / 10;
    int r1 = csc2[l] % 10;
    if (MEPath[1][s1][r1] && MEPath[0][s1][r1]) {
      double dME_phi = ME_phi[1][s1][r1] - ME_phi[0][s1][r1];
      double dME_eta = ME_eta[1][s1][r1] - ME_eta[0][s1][r1];
      hME2[l]->Fill8b((pt1[0] * ME_phi[0][s1][r1]), ME_phi[0][s1][r1], ME_eta[0][s1][r1], pt1[0]);
      hME2[l]->Fill8c((pt1[0] * ME_phi[1][s1][r1]), ME_phi[1][s1][r1], dME_phi, dME_eta, ME_eta[1][s1][r1], pt1[0]);
    }
  }
}

void MuonSeedParaFillHisto::FillDTSegmentSingle(
    H2DRecHit7* hMB2[12], double pt1[5], double MB_phi[2][5][3], double MB_eta[2][5][3], bool MBPath[2][5][3]) {
  int dt2[12] = {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42};
  for (int l = 0; l < 12; l++) {
    int s1 = dt2[l] / 10;
    int w1 = dt2[l] % 10;
    if (MBPath[1][s1][w1] && MBPath[0][s1][w1]) {
      double dMB_phi = MB_phi[1][s1][w1] - MB_phi[0][s1][w1];
      double dMB_eta = MB_eta[1][s1][w1] - MB_eta[0][s1][w1];
      hMB2[l]->Fill9b((pt1[0] * MB_phi[0][s1][w1]), MB_phi[0][s1][w1], MB_eta[0][s1][w1], pt1[0]);
      hMB2[l]->Fill9c((pt1[0] * MB_phi[1][s1][w1]), MB_phi[1][s1][w1], dMB_phi, dMB_eta, MB_eta[1][s1][w1], pt1[0]);
    }
  }
}
/// For reco-segment treea
/*
  tt = tr_muon;
  if ( MEPath[1][1] && MEPath[1][2] && MEPath[1][3] ) {
      tt->Fill_b1(fabs(EtaP1[1][1]),fabs(EtaP1[1][2]),fabs(EtaP1[1][3]),fabs(EtaP1[1][4]), 
                  EtaP1[1][1], EtaP1[1][2], EtaP1[1][3], EtaP1[1][4], pt1[0]);
      tt->Fill_l1(pa[0]);
  }
  if ( MBPath[1][1] && MBPath[1][2] && MBPath[1][3] ) {
      tt->Fill_b2(fabs(EtaP3[1][1]),fabs(EtaP3[1][2]),fabs(EtaP3[1][3]),fabs(EtaP3[1][4]), 
                  EtaP3[1][1], EtaP3[1][2], EtaP3[1][3], EtaP3[1][4], pt1[0]);
      tt->Fill_l1(pa[0]);
  }
  tt->FillTree();
  */
