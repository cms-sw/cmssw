#include "Alignment/MuonAlignmentAlgorithms/interface/DTMuonSLToSL.h"


DTMuonSLToSL::DTMuonSLToSL(std::string path, int n_files, float MaxPt, float MinPt, TFile *f_) {

  ntuplePath = path;
  numberOfRootFiles = n_files;
 
  f = f_;  

  ptMax = MaxPt; 
  ptMin = MinPt; 

  setBranchTree();

  initNTuples(0);

  calculationSLToSL();
  
}

DTMuonSLToSL::~DTMuonSLToSL() {}



void DTMuonSLToSL::calculationSLToSL() {

  
  TMatrixD ****C13 = new TMatrixD ***[5];
  TMatrixD ****b13 = new TMatrixD ***[5];
  TMatrixD ****C31 = new TMatrixD ***[5];
  TMatrixD ****b31 = new TMatrixD ***[5];
  
  for(int whI = -2; whI < 3; ++whI) {
    C13[whI+2] = new TMatrixD **[4];
    b13[whI+2] = new TMatrixD **[4];
    C31[whI+2] = new TMatrixD **[4];
    b31[whI+2] = new TMatrixD **[4];
    for(int stI = 1; stI < 5; ++stI) {
      C13[whI+2][stI-1] = new TMatrixD * [14];
      b13[whI+2][stI-1] = new TMatrixD * [14];
      C31[whI+2][stI-1] = new TMatrixD * [14];
      b31[whI+2][stI-1] = new TMatrixD * [14];
      for(int seI = 1; seI < 15; ++seI) {
        if(seI > 12 && stI != 4) continue;
        C13[whI+2][stI-1][seI-1] = new TMatrixD(3,3);
        b13[whI+2][stI-1][seI-1] = new TMatrixD(3,1);
        C31[whI+2][stI-1][seI-1] = new TMatrixD(3,3);
        b31[whI+2][stI-1][seI-1] = new TMatrixD(3,1);
      }
    }
  }
  
  //Run over the TTree  
  Int_t nentries = (Int_t)tali->GetEntries();
  for (Int_t i=0;i<nentries;i++) {
    tali->GetEntry(i);
    //Basic cuts
    if(pt > ptMax || pt < ptMin) continue;
   
    bool repeatedHits = false;  
    for(int counter = 0; counter < nseg; ++counter) {
      //Make sure there are no repeated hits
      for(int counterHi = 0; counterHi < nhits[counter]; counterHi++) {
        for(int counterHj = 0; counterHj < nhits[counter]; counterHj++) {
          if(counterHi == counterHj) continue;
          if(zc[counter][counterHi] == zc[counter][counterHj]) {
            repeatedHits = true;
          }
        }
      }
      if(repeatedHits == true) continue;
          
      float x_13 = xSlSL3[counter]; float xp_13 = xSL1SL3[counter];
      float x_31 = xSlSL1[counter]; float xp_31 = xSL3SL1[counter];
      //float tanphi = dxdzSl[counter];
      float tanphi_13 = dxdzSlSL1[counter];
      float tanphi_31 = dxdzSlSL3[counter];
      int wheel = wh[counter];
      int station = st[counter];
      int sector = sr[counter];

      if(fabs(x_13-xp_13)< 3 && fabs(x_31-xp_31) && fabs(tanphi_13-tanphi_31)<0.06) {
        
	*(C13[wheel+2][station-1][sector-1]) += returnCSLMatrix(x_13, xp_13, tanphi_13);
	*(b13[wheel+2][station-1][sector-1]) += returnbSLMatrix(x_13, xp_13, tanphi_13);
	
        *(C31[wheel+2][station-1][sector-1]) += returnCSLMatrix(x_31, xp_31, tanphi_31);
	*(b31[wheel+2][station-1][sector-1]) += returnbSLMatrix(x_31, xp_31, tanphi_31);
      }
    }
  }

  for(int wheel = -2; wheel < 3; ++wheel) {
    for(int station = 1; station < 5; ++station) {
      for(int sector = 1; sector < 15; ++sector) {
        if(sector > 12 && station != 4) continue;
        TMatrixD solution13(3,1);
        TMatrixD solution31(3,1);
        TMatrixD C31_copy = *(C31[wheel+2][station-1][sector-1]);
        TMatrixD C13_copy = *(C13[wheel+2][station-1][sector-1]);
        TMatrixD b31_copy = *(b31[wheel+2][station-1][sector-1]);
        TMatrixD b13_copy = *(b13[wheel+2][station-1][sector-1]);

        C31_copy.Invert();
        C13_copy.Invert();
        solution13 = C13_copy * b13_copy;
        solution31 = C31_copy * b31_copy;
        whC = wheel; stC = station; srC = sector;
        dx = solution13(0,0); dz = solution13(1,0);
        phiy = solution13(2,0);
        for(int c = 0; c < 3; ++c) {
          for(int s = 0; s < 3; ++s) {
            cov[c][s] = C13_copy(c, s);
          }
        } 
        ttreeOutput->Fill();
      }
    }
  }
  f->Write();
}



TMatrixD DTMuonSLToSL::returnCSLMatrix(float x, float xp, float tanphi) {

  TMatrixD matrix(3,3);

  matrix(0,0) = 1.0;
  matrix(1,0) = tanphi;
  matrix(0,1) = tanphi;
  matrix(1,1) = tanphi*tanphi;
  matrix(0,2) = tanphi*xp;
  matrix(2,0) = tanphi*xp;
  matrix(2,2) = tanphi*tanphi*xp*xp;
  matrix(2,1) = tanphi*tanphi*xp;
  matrix(1,2) = tanphi*tanphi*xp;

  return matrix;

}
 

TMatrixD DTMuonSLToSL::returnbSLMatrix(float x, float xp, float tanphi) {

  TMatrixD matrix(3,1);

  matrix(0,0) = -(x-xp);
  matrix(1,0) = -(x-xp)*tanphi;
  matrix(2,0) = -(x-xp)*tanphi*xp;

  return matrix;

}


void DTMuonSLToSL::setBranchTree() {

  ttreeOutput = new TTree("DTSLToSLResult", "DTSLToSLResult");

  ttreeOutput->Branch("wh", &whC, "wh/F");
  ttreeOutput->Branch("st", &stC, "st/F");
  ttreeOutput->Branch("sr", &srC, "sr/F");
  ttreeOutput->Branch("dx", &dx, "dx/F");
  ttreeOutput->Branch("dz", &dz, "dz/F");
  ttreeOutput->Branch("phiy", &phiy, "phiy/F");
  ttreeOutput->Branch("cov", cov, "cov[3][3]/F");

}



