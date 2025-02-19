#ifndef ElectronMVA_H
#define ElectronMVA_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "TMVA/Reader.h"
#include<string>

class ElectronMVAEstimator {
 public:
  ElectronMVAEstimator();
  ElectronMVAEstimator(std::string fileName);
  ~ElectronMVAEstimator() {;}
  double mva(const reco::GsfElectron& myElectron, int nvertices=0);

 private:
  void bindVariables();
  void init(std::string fileName);

 private:
  TMVA::Reader    *tmvaReader_;
  
  Float_t       fbrem;
  Float_t       detain;
  Float_t       dphiin; 
  Float_t       sieie;
  Float_t       hoe;
  Float_t       eop;
  Float_t       e1x5e5x5;
  Float_t       eleopout;
  Float_t       detaeleout;
  Float_t       kfchi2;
  Float_t       dist;
  Float_t       dcot;
  Float_t       eta;
  Float_t       pt;
  Int_t         kfhits;
  Int_t         mishits;
  Int_t         ecalseed;
  Int_t         Nvtx;

  Float_t       absdist;
  Float_t       absdcot;
  Float_t       mykfhits;
  Float_t       mymishits;
  Float_t       myNvtx;
};

#endif
