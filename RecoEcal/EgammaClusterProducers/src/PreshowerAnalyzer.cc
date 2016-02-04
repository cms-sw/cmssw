/**\class PreshowerAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Shahram Rahatlou
//         Created:  10 May 200
// $Id: PreshowerAnalyzer.cc,v 1.6 2009/12/18 20:45:01 wmtan Exp $
//

#include "RecoEcal/EgammaClusterProducers/interface/PreshowerAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TFile.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"

//========================================================================
PreshowerAnalyzer::PreshowerAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{
 
  EminDE_ = ps.getParameter<double>("EminDE");
  EmaxDE_ = ps.getParameter<double>("EmaxDE");
  nBinDE_ = ps.getParameter<int>("nBinDE");

  EminSC_ = ps.getParameter<double>("EminSC");
  EmaxSC_ = ps.getParameter<double>("EmaxSC");
  nBinSC_ = ps.getParameter<int>("nBinSC");

  preshClusterCollectionX_ = ps.getParameter<std::string>("preshClusterCollectionX");
  preshClusterCollectionY_ = ps.getParameter<std::string>("preshClusterCollectionY");
  preshClusterProducer_   = ps.getParameter<std::string>("preshClusterProducer");

  islandEndcapSuperClusterCollection1_ = ps.getParameter<std::string>("islandEndcapSuperClusterCollection1");
  islandEndcapSuperClusterProducer1_   = ps.getParameter<std::string>("islandEndcapSuperClusterProducer1");

  islandEndcapSuperClusterCollection2_ = ps.getParameter<std::string>("islandEndcapSuperClusterCollection2");
  islandEndcapSuperClusterProducer2_   = ps.getParameter<std::string>("islandEndcapSuperClusterProducer2");

  outputFile_   = ps.getParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms

  // calibration parameters:
  calib_planeX_ = ps.getParameter<double>("preshCalibPlaneX");
  calib_planeY_ = ps.getParameter<double>("preshCalibPlaneY");
  gamma_        = ps.getParameter<double>("preshCalibGamma");
  mip_          = ps.getParameter<double>("preshCalibMIP");

  nEvt_ = 0; 
}

//========================================================================
PreshowerAnalyzer::~PreshowerAnalyzer()
//========================================================================
{
  delete rootFile_;
}

//========================================================================
void PreshowerAnalyzer::beginJob() {
//========================================================================

  rootFile_->cd();

  h1_esE_x = new TH1F("esE_x"," ES cluster Energy in  X-plane",20, 0, 0.03);
  h1_esE_y = new TH1F("esE_y"," ES cluster Energy in  Y-plane",20, 0, 0.03);
  h1_esEta_x = new TH1F("esEta_x"," ES cluster Eta in X-plane",12, 1.5, 2.7);
  h1_esEta_y = new TH1F("esEta_y"," ES cluster Eta in Y-plane",12, 1.5, 2.7);
  h1_esPhi_x = new TH1F("esPhi_x"," ES cluster Phi in X-plane",20, 0, 6.28);
  h1_esPhi_y = new TH1F("esPhi_y"," ES cluster Phi in Y-plane",20, 0, 6.28);
  h1_esNhits_x = new TH1F("esNhits_x"," ES cluster Nhits in  X-plane",10, 0, 10);
  h1_esNhits_y = new TH1F("esNhits_y"," ES cluster Nhits in  Y-plane",10, 0, 10);
  h1_esDeltaE = new TH1F("esDeltaE"," DeltaE", nBinDE_, EminDE_, EmaxDE_); 
  h1_nclu_x = new TH1F("esNclu_x"," number of ES clusters (for one SC) in X-plane",20, 0, 80);
  h1_nclu_y = new TH1F("esNclu_y"," number of ES clusters (for one SC) in Y-plane",20, 0, 80);

  h1_islandEESCEnergy1 = new TH1F("islandEESCEnergy1","Energy of super clusters with island algo - endcap1",nBinSC_,EminSC_,EmaxSC_);
  h1_islandEESCEnergy2 = new TH1F("islandEESCEnergy2","Energy of super clusters with island algo - endcap2",nBinSC_,EminSC_,EmaxSC_);
}


//========================================================================
void
PreshowerAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classe

  //std::cout << "\n .......  Event # " << nEvt_+1 << " is analyzing ....... " << std::endl << std::endl;

  // Get island super clusters
  Handle<reco::SuperClusterCollection> pIslandEndcapSuperClusters1;
  evt.getByLabel(islandEndcapSuperClusterProducer1_, islandEndcapSuperClusterCollection1_, pIslandEndcapSuperClusters1);
  const reco::SuperClusterCollection* islandEndcapSuperClusters1 = pIslandEndcapSuperClusters1.product();
  //std::cout << "\n islandEndcapSuperClusters1->size() = " << islandEndcapSuperClusters1->size() << std::endl;

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = islandEndcapSuperClusters1->begin();
                                                    aClus != islandEndcapSuperClusters1->end(); aClus++) {
    h1_islandEESCEnergy1->Fill( aClus->energy() );
  }

  // Get island super clusters
  Handle<reco::SuperClusterCollection> pIslandEndcapSuperClusters2;
  evt.getByLabel(islandEndcapSuperClusterProducer2_, islandEndcapSuperClusterCollection2_, pIslandEndcapSuperClusters2);
  const reco::SuperClusterCollection* islandEndcapSuperClusters2 = pIslandEndcapSuperClusters2.product();
  //std::cout << "\n islandEndcapSuperClusters2->size() = " << islandEndcapSuperClusters2->size() << std::endl;

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = islandEndcapSuperClusters2->begin();
                                                    aClus != islandEndcapSuperClusters2->end(); aClus++) {
    h1_islandEESCEnergy2->Fill( aClus->energy() );
  }


  // Get ES clusters in X plane
  Handle<reco::PreshowerClusterCollection> pPreshowerClustersX;
  evt.getByLabel(preshClusterProducer_, preshClusterCollectionX_, pPreshowerClustersX);
  const reco::PreshowerClusterCollection *clustersX = pPreshowerClustersX.product();
  h1_nclu_x->Fill( clustersX->size() );
  //std::cout << "\n pPreshowerClustersX->size() = " << clustersX->size() << std::endl;

  Handle<reco::PreshowerClusterCollection> pPreshowerClustersY;
  evt.getByLabel(preshClusterProducer_, preshClusterCollectionY_, pPreshowerClustersY);
  const reco::PreshowerClusterCollection *clustersY = pPreshowerClustersY.product();
  h1_nclu_y->Fill( clustersY->size() );
  //std::cout << "\n pPreshowerClustersY->size() = " << clustersY->size() << std::endl;


  // loop over the ES clusters and fill the histogram
  float e1 = 0;
  for(reco::PreshowerClusterCollection::const_iterator esClus = clustersX->begin();
                                                       esClus !=clustersX->end(); esClus++) {
      e1 += esClus->energy();
      h1_esE_x->Fill( esClus->energy() );  
      h1_esEta_x->Fill( esClus->eta() );
      h1_esPhi_x->Fill( esClus->phi() );
      h1_esNhits_x->Fill( esClus->nhits() );     
  }

  float e2 = 0;
  for(reco::PreshowerClusterCollection::const_iterator esClus = clustersY->begin();
                                                       esClus !=clustersY->end(); esClus++) {
      e2 += esClus->energy();
      h1_esE_y->Fill( esClus->energy() );  
      h1_esEta_y->Fill( esClus->eta() );
      h1_esPhi_y->Fill( esClus->phi() );
      h1_esNhits_y->Fill( esClus->nhits() );     
  }
  
  float deltaE = 0;       
  if(e1+e2 > 1.0e-10) {
      // GeV to #MIPs
      e1 = e1 / mip_;
      e2 = e2 / mip_;
      deltaE = gamma_*(calib_planeX_*e1+calib_planeY_*e2);       
   }

  h1_esDeltaE->Fill(deltaE);

   nEvt_++;

}

//========================================================================
void PreshowerAnalyzer::endJob() {
//========================================================================

   rootFile_->cd();

   h1_esE_x->Write();     
   h1_esE_y->Write();
   h1_esEta_x->Write();
   h1_esEta_y->Write();
   h1_esPhi_x->Write();
   h1_esPhi_y->Write();
   h1_esNhits_x->Write();
   h1_esNhits_y->Write();          
   h1_esDeltaE->Write();
   h1_nclu_x->Write();
   h1_nclu_y->Write();

   h1_islandEESCEnergy1->Write();
   h1_islandEESCEnergy2->Write();

   rootFile_->Close();
}
