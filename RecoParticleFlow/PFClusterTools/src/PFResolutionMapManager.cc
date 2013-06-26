#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMapManager.h"
#include <iostream>
#include <TFile.h>

using namespace std;

PFResolutionMapManager::PFResolutionMapManager(const char * mapfile){
  TFile f(mapfile);
  TH2D *hSEtaC = (TH2D*)f.Get("Sigma_Eta_WithECorrection");
  TH2D *hSEta = (TH2D*)f.Get("Sigma_Eta");
  TH2D *hSPhiC = (TH2D*)f.Get("Sigma_Phi_WithECorrection");
  TH2D *hSPhi = (TH2D*)f.Get("Sigma_Phi");

  M1 = PFResolutionMap(*hSEtaC);
  M2 = PFResolutionMap(*hSEta);
  M3 = PFResolutionMap(*hSPhiC);
  M4 = PFResolutionMap(*hSPhi);
}

const PFResolutionMap& PFResolutionMapManager::GetResolutionMap(bool MapEta,bool Corr){
  if(MapEta){
    if(Corr) return M1;
    else return M2;
  }
  else{
    if(Corr) return M3;
    else return M4;
  }
}
