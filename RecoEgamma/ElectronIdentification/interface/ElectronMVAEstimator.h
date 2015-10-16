#ifndef __RecoEgamma_ElectronIdentification_ElectronMVAEstimator_H__
#define __RecoEgamma_ElectronIdentification_ElectronMVAEstimator_H__

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <memory>
#include <string>

class ElectronMVAEstimator {
 public:
  struct Configuration{
         std::vector<std::string> vweightsfiles;
   };
  ElectronMVAEstimator();
  ElectronMVAEstimator(std::string fileName);
  ElectronMVAEstimator(const Configuration & );
  ~ElectronMVAEstimator() {;}
  double mva(const reco::GsfElectron& myElectron, int nvertices=0) const;

 private:
  const Configuration cfg_;
  void bindVariables(float vars[18]) const;
  
  std::vector<std::unique_ptr<const GBRForest> > gbr;
  
  Float_t       fbrem;      //0
  Float_t       detain;     //1
  Float_t       dphiin;     //2
  Float_t       sieie;      //3
  Float_t       hoe;        //4
  Float_t       eop;        //5
  Float_t       e1x5e5x5;   //6
  Float_t       eleopout;   //7
  Float_t       detaeleout; //8
  Float_t       kfchi2;     //9
  Float_t       mykfhits;   //10
  Float_t       mymishits;  //11
  Float_t       absdist;    //12
  Float_t       absdcot;    //13
  Float_t       myNvtx;     //14
  Float_t       eta;        //15
  Float_t       pt;         //16
  Int_t         ecalseed;   //17
};

#endif
