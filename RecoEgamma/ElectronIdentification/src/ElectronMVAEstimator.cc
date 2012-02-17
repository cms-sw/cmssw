#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

ElectronMVAEstimator::ElectronMVAEstimator(){}

ElectronMVAEstimator::ElectronMVAEstimator(std::string fileName){
  init(fileName);
}


void ElectronMVAEstimator::init(std::string fileName) {
  tmvaReader_ = new TMVA::Reader("!Color:Silent");
  tmvaReader_->AddVariable("fbrem",&fbrem);
  tmvaReader_->AddVariable("detain", &detain);
  tmvaReader_->AddVariable("dphiin", &dphiin);
  tmvaReader_->AddVariable("sieie", &sieie);
  tmvaReader_->AddVariable("hoe", &hoe);
  tmvaReader_->AddVariable("eop", &eop);
  tmvaReader_->AddVariable("e1x5e5x5", &e1x5e5x5);
  tmvaReader_->AddVariable("eleopout", &eleopout);
  tmvaReader_->AddVariable("detaeleout", &detaeleout);
  tmvaReader_->AddVariable("kfchi2", &kfchi2);
  tmvaReader_->AddVariable("kfhits", &mykfhits);
  tmvaReader_->AddVariable("mishits",&mymishits);
  tmvaReader_->AddVariable("dist", &absdist);
  tmvaReader_->AddVariable("dcot", &absdcot);
  tmvaReader_->AddVariable("nvtx", &myNvtx);

  tmvaReader_->AddSpectator("eta",&eta);
  tmvaReader_->AddSpectator("pt",&pt);
  tmvaReader_->AddSpectator("ecalseed",&ecalseed);
  
  // Taken from Daniele (his mail from the 30/11)
  //  tmvaReader_->BookMVA("BDTSimpleCat","../Training/weights_Root527b_3Depth_DanVarConvRej_2PtBins_10Pt_800TPrune5_Min100Events_NoBjets_half/TMVA_BDTSimpleCat.weights.xm");
  // training of the 7/12 with Nvtx added
  tmvaReader_->BookMVA("BDTSimpleCat",fileName.c_str());
}



double ElectronMVAEstimator::mva(const reco::GsfElectron& myElectron, int nvertices )  {
  fbrem = myElectron.fbrem();
  detain = myElectron.deltaEtaSuperClusterTrackAtVtx();
  dphiin = myElectron.deltaPhiSuperClusterTrackAtVtx();
  sieie = myElectron.sigmaIetaIeta();
  hoe = myElectron.hcalOverEcal();
  eop = myElectron.eSuperClusterOverP();
  e1x5e5x5 = (myElectron.e5x5()) !=0. ? 1.-(myElectron.e1x5()/myElectron.e5x5()) : -1. ;
  eleopout = myElectron.eEleClusterOverPout();
  detaeleout = myElectron.deltaEtaEleClusterTrackAtCalo();
  
  bool validKF= false;

  reco::TrackRef myTrackRef = myElectron.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());  

  kfchi2 = (validKF) ? myTrackRef->normalizedChi2() : 0 ;
  kfhits = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ; 
  dist = myElectron.convDist();
  dcot = myElectron.convDcot();
  eta = myElectron.eta();
  pt = myElectron.pt();
  
  mishits = myElectron.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits();
  ecalseed = myElectron.ecalDrivenSeed();
  
  Nvtx = nvertices;

  bindVariables();
  double result =  tmvaReader_->EvaluateMVA("BDTSimpleCat");
//  
//  std::cout << "fbrem" <<fbrem << std::endl;
//  std::cout << "detain"<< detain << std::endl;
//  std::cout << "dphiin"<< dphiin << std::endl;
//  std::cout << "sieie"<< sieie << std::endl;
//  std::cout << "hoe"<< hoe << std::endl;
//  std::cout << "eop"<< eop << std::endl;
//  std::cout << "e1x5e5x5"<< e1x5e5x5 << std::endl;
//  std::cout << "eleopout"<< eleopout << std::endl;
//  std::cout << "detaeleout"<< detaeleout << std::endl;
//  std::cout << "kfchi2"<< kfchi2 << std::endl;
//  std::cout << "kfhits"<< mykfhits << std::endl;
//  std::cout << "mishits"<<mymishits << std::endl;
//  std::cout << "dist"<< absdist << std::endl;
//  std::cout << "dcot"<< absdcot << std::endl;
//
//  std::cout << "eta"<<eta << std::endl;
//  std::cout << "pt"<<pt << std::endl;
//  std::cout << "ecalseed"<<ecalseed << std::endl;
//
//  std::cout << " MVA " << result << std::endl;
  return result;
}


void ElectronMVAEstimator::bindVariables() {
  if(fbrem < -1.)
    fbrem = -1.;  
  
  detain = fabs(detain);
  if(detain > 0.06)
    detain = 0.06;
  
  
  dphiin = fabs(dphiin);
  if(dphiin > 0.6)
    dphiin = 0.6;

  
  if(eop > 20.)
    eop = 20.;
  
  
  if(eleopout > 20.)
    eleopout = 20;
  
  detaeleout = fabs(detaeleout);
  if(detaeleout > 0.2)
    detaeleout = 0.2;
  
  mykfhits = float(kfhits);
  mymishits = float(mishits);
  
  if(kfchi2 < 0.)
    kfchi2 = 0.;
  
  if(kfchi2 > 15.)
    kfchi2 = 15.;
  
  
  if(e1x5e5x5 < -1.)
    e1x5e5x5 = -1;

  if(e1x5e5x5 > 2.)
    e1x5e5x5 = 2.; 
  
  
  if(dist > 15.)
    dist = 15.;
  if(dist < -15.)
    dist = -15.;
  
  if(dcot > 3.)
    dcot = 3.;
  if(dcot < -3.)
    dcot = -3.;
  
  absdist = fabs(dist);
  absdcot = fabs(dcot);
  myNvtx = float(Nvtx);

}
