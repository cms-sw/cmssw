#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

ElectronMVAEstimator::ElectronMVAEstimator():
  cfg_{}
{}

ElectronMVAEstimator::ElectronMVAEstimator(const std::string& fileName):
  cfg_{} 
{
  // Taken from Daniele (his mail from the 30/11)
  //  tmvaReader.BookMVA("BDTSimpleCat","../Training/weights_Root527b_3Depth_DanVarConvRej_2PtBins_10Pt_800TPrune5_Min100Events_NoBjets_half/TMVA_BDTSimpleCat.weights.xm");
  // training of the 7/12 with Nvtx added
  gbr_.push_back( createGBRForest(fileName) );
}

ElectronMVAEstimator::ElectronMVAEstimator(const Configuration & cfg):cfg_(cfg)
{
  for(const auto& weightsfile : cfg_.vweightsfiles) {
    gbr_.push_back( createGBRForest(weightsfile) );
  }
}

double ElectronMVAEstimator::mva(const reco::GsfElectron& myElectron, int nvertices ) const {
  float vars[18];

  vars[0] = myElectron.fbrem();
  vars[1] = std::abs(myElectron.deltaEtaSuperClusterTrackAtVtx());
  vars[2] = std::abs(myElectron.deltaPhiSuperClusterTrackAtVtx());
  vars[3] = myElectron.sigmaIetaIeta();
  vars[4] = myElectron.hcalOverEcal();
  vars[5] = myElectron.eSuperClusterOverP();
  vars[6] = (myElectron.e5x5()) !=0. ? 1.-(myElectron.e1x5()/myElectron.e5x5()) : -1. ;
  vars[7] = myElectron.eEleClusterOverPout();
  vars[8] = std::abs(myElectron.deltaEtaEleClusterTrackAtCalo());
  
  bool validKF= false;

  reco::TrackRef myTrackRef = myElectron.closestCtfTrackRef();
  validKF = (myTrackRef.isNonnull() && myTrackRef.isAvailable());   

  vars[9] = (validKF) ? myTrackRef->normalizedChi2() : 0 ;
  vars[10] = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1.; 
  vars[11] = myElectron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  vars[12] = std::abs(myElectron.convDist());
  vars[13] = std::abs(myElectron.convDcot());
  vars[14] = nvertices;
  vars[15] = myElectron.eta();
  vars[16] = myElectron.pt();
  vars[17] = myElectron.ecalDrivenSeed();
  
  bindVariables(vars);

  //0 pt &lt; 10 &amp;&amp; abs(eta)&lt;=1.485
  //1 pt &gt;= 10 &amp;&amp; abs(eta)&lt;=1.485
  //2 pt &lt; 10 &amp;&amp; abs(eta)&gt; 1.485
  //3 pt &gt;= 10 &amp;&amp;  abs(eta)&gt; 1.485

  const unsigned index = (unsigned)(myElectron.pt() >= 10) + 2*(unsigned)(std::abs(myElectron.eta()) > 1.485);

  double result =  gbr_[index]->GetAdaBoostClassifier(vars);
//  
//  std::cout << "fbrem" << vars[0] << std::endl;
//  std::cout << "detain"<< vars[1] << std::endl;
//  std::cout << "dphiin"<< vars[2] << std::endl;
//  std::cout << "sieie"<< vars[3] << std::endl;
//  std::cout << "hoe"<< vars[4] << std::endl;
//  std::cout << "eop"<< vars[5] << std::endl;
//  std::cout << "e1x5e5x5"<< vars[6] << std::endl;
//  std::cout << "eleopout"<< vars[7] << std::endl;
//  std::cout << "detaeleout"<< vars[8] << std::endl;
//  std::cout << "kfchi2"<< vars[9] << std::endl;
//  std::cout << "kfhits"<< vars[10] << std::endl;
//  std::cout << "mishits"<<vars[11] << std::endl;
//  std::cout << "dist"<< vars[12] << std::endl;
//  std::cout << "dcot"<< vars[13] << std::endl;
//  std::cout << "nvtx"<< vars[14] << std::endl;
//  std::cout << "eta"<< vars[15] << std::endl;
//  std::cout << "pt"<< vars[16] << std::endl;
//  std::cout << "ecalseed"<< vars[17] << std::endl;
//
//  std::cout << " MVA " << result << std::endl;
  return result;
}


void ElectronMVAEstimator::bindVariables(float vars[18]) const {
  if(vars[0] < -1.)
    vars[1] = -1.;  
  
  if(vars[1] > 0.06)
    vars[1] = 0.06;
    
  if(vars[2] > 0.6)
    vars[2] = 0.6;
  
  if(vars[5] > 20.)
    vars[5] = 20.;
    
  if(vars[7] > 20.)
    vars[7] = 20;
  
  if(vars[8] > 0.2)
    vars[8] = 0.2;
  
  if(vars[9] < 0.)
    vars[9] = 0.;
  
  if(vars[9] > 15.)
    vars[9] = 15.;
    
  if(vars[6] < -1.)
    vars[6] = -1;

  if(vars[6] > 2.)
    vars[6] = 2.; 
    
  if(vars[12] > 15.)
    vars[12] = 15.;
    
  if(vars[13] > 3.)
    vars[13] = 3.;
}
