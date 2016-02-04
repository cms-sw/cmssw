/**\class EgammaSimpleAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Shahram Rahatlou
//         Created:  10 May 2006
// $Id: EgammaSimpleAnalyzer.cc,v 1.14 2009/12/18 20:45:01 wmtan Exp $
//

#include "RecoEcal/EgammaClusterProducers/interface/EgammaSimpleAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TFile.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"


//========================================================================
EgammaSimpleAnalyzer::EgammaSimpleAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{

  xMinHist_ = ps.getParameter<double>("xMinHist");
  xMaxHist_ = ps.getParameter<double>("xMaxHist");
  nbinHist_ = ps.getParameter<int>("nbinHist");

  islandBarrelBasicClusterCollection_ = ps.getParameter<std::string>("islandBarrelBasicClusterCollection");
  islandBarrelBasicClusterProducer_   = ps.getParameter<std::string>("islandBarrelBasicClusterProducer");
  islandBarrelBasicClusterShapes_   = ps.getParameter<std::string>("islandBarrelBasicClusterShapes");

  islandEndcapBasicClusterCollection_ = ps.getParameter<std::string>("islandEndcapBasicClusterCollection");
  islandEndcapBasicClusterProducer_   = ps.getParameter<std::string>("islandEndcapBasicClusterProducer");
  islandEndcapBasicClusterShapes_   = ps.getParameter<std::string>("islandEndcapBasicClusterShapes");

  islandEndcapSuperClusterCollection_ = ps.getParameter<std::string>("islandEndcapSuperClusterCollection");
  islandEndcapSuperClusterProducer_   = ps.getParameter<std::string>("islandEndcapSuperClusterProducer");

  correctedIslandEndcapSuperClusterCollection_ = ps.getParameter<std::string>("correctedIslandEndcapSuperClusterCollection");
  correctedIslandEndcapSuperClusterProducer_   = ps.getParameter<std::string>("correctedIslandEndcapSuperClusterProducer");

  hybridSuperClusterCollection_ = ps.getParameter<std::string>("hybridSuperClusterCollection");
  hybridSuperClusterProducer_   = ps.getParameter<std::string>("hybridSuperClusterProducer");

  correctedHybridSuperClusterCollection_ = ps.getParameter<std::string>("correctedHybridSuperClusterCollection");
  correctedHybridSuperClusterProducer_   = ps.getParameter<std::string>("correctedHybridSuperClusterProducer");

  outputFile_   = ps.getParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms

}


//========================================================================
EgammaSimpleAnalyzer::~EgammaSimpleAnalyzer()
//========================================================================
{

  // apparently ROOT takes ownership of histograms
  // created after opening a new TFile... no delete is needed
  // ... mysteries of root...
  /*
  delete h1_islandBCEnergy_;
  delete h1_islandSCEnergy_;
  delete h1_corrIslandSCEnergy_;
  delete h1_hybridSCEnergy_;
  delete h1_corrHybridSCEnergy_;
  delete h1_corrHybridSCEta_;
  delete h1_corrHybridSCPhi_;
  */
  delete rootFile_;
}

//========================================================================
void
EgammaSimpleAnalyzer::beginJob() {
//========================================================================

  // go to *OUR* rootfile and book histograms
  rootFile_->cd();

  h1_nIslandEBBC_ = new TH1F("nIslandEBBC","# basic clusters with island in barrel",11,-0.5,10.5);
  h1_nIslandEEBC_ = new TH1F("nIslandEEBC","# basic clusters with island in endcap",11,-0.5,10.5);

  h1_nIslandEESC_ = new TH1F("nIslandEESC","# super clusters with island in endcap",11,-0.5,10.5);
  h1_nHybridSC_ = new TH1F("nHybridSC","# super clusters with hybrid",11,-0.5,10.5);

  h1_islandEBBCEnergy_ = new TH1F("islandEBBCEnergy","Energy of basic clusters with island algo - barrel",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandEBBCXtals_ = new TH1F("islandEBBCXtals","#xtals in basic cluster - island barrel",51,-0.5,50.5);

  h1_islandEBBCe9over25_= new TH1F("islandEBBCe9over25","e3x3/e5x5 of basic clusters with island algo - barrel",35,0.5,1.2);
  h1_islandEBBCe5x5_ = new TH1F("islandEBBCe5x5","e5x5 of basic clusters with island algo - barrel",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandEEBCe5x5_ = new TH1F("islandEEBCe5x5","e5x5 of basic clusters with island algo - endcap",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandEEBCEnergy_ = new TH1F("islandEEBCEnergy","Energy of basic clusters with island algo - endcap",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandEEBCXtals_ = new TH1F("islandEEBCXtals","#xtals in basic cluster - island endcap",51,-0.5,50.5);

  h1_islandEESCEnergy_ = new TH1F("islandEESCEnergy","Energy of super clusters with island algo - endcap",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrIslandEESCEnergy_ = new TH1F("corrIslandEESCEnergy","Corrected Energy of super clusters with island algo - endcap",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrIslandEESCET_ = new TH1F("corrIslandEESCET","Corrected Transverse Energy of super clusters with island algo - endcap",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandEESCClusters_ = new TH1F("islandEESCClusters","# basic clusters in super cluster - island endcap",11,-0.5,10.5);

  h1_hybridSCEnergy_ = new TH1F("hybridSCEnergy","Energy of super clusters with hybrid algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrHybridSCEnergy_ = new TH1F("corrHybridSCEnergy","Corrected Energy of super clusters with hybrid algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrHybridSCET_ = new TH1F("corrHybridSCET","Corrected Transverse Energy of super clusters with hybrid algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrHybridSCEta_ = new TH1F("corrHybridSCEta","Eta of super clusters with hybrid algo",40,-3.,3.);
  h1_corrHybridSCPhi_ = new TH1F("corrHybridSCPhi","Phi of super clusters with hybrid algo",40,0.,6.28);
  h1_hybridSCClusters_ = new TH1F("hybridSCClusters","# basic clusters in super cluster - hybrid",11,-0.5,10.5);

}


//========================================================================
void
EgammaSimpleAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classes


  //  ----- barrel with island

  // Get island basic clusters
  Handle<reco::BasicClusterCollection> pIslandBarrelBasicClusters;
  evt.getByLabel(islandBarrelBasicClusterProducer_, islandBarrelBasicClusterCollection_, pIslandBarrelBasicClusters);
  const reco::BasicClusterCollection* islandBarrelBasicClusters = pIslandBarrelBasicClusters.product();
  h1_nIslandEBBC_->Fill(islandBarrelBasicClusters->size());

  // fetch cluster shapes of island basic clusters in barrel
  Handle<reco::ClusterShapeCollection> pIslandEBShapes;
  evt.getByLabel(islandBarrelBasicClusterProducer_, islandBarrelBasicClusterShapes_, pIslandEBShapes);
  const reco::ClusterShapeCollection* islandEBShapes = pIslandEBShapes.product();

  std::ostringstream str;
  str << "# island basic clusters in barrel: " << islandBarrelBasicClusters->size()
      << "\t# associated cluster shapes: " << islandEBShapes->size() << "\n"
      << "Loop over island basic clusters in barrel" << "\n";

  // loop over the Basic clusters and fill the histogram
  int iClus=0;
  for(reco::BasicClusterCollection::const_iterator aClus = islandBarrelBasicClusters->begin();
                                                    aClus != islandBarrelBasicClusters->end(); aClus++) {
    h1_islandEBBCEnergy_->Fill( aClus->energy() );
    h1_islandEBBCXtals_->Fill(  aClus->size() );
    str << "energy: " << aClus->energy()
        << "\te5x5: " << (*islandEBShapes)[iClus].e5x5()
        << "\te2x2: " << (*islandEBShapes)[iClus].e2x2()
        << "\n";
    h1_islandEBBCe5x5_->Fill( (*islandEBShapes)[iClus].e5x5() );

    iClus++;
  }
  edm::LogInfo("EgammaSimpleAnalyzer") << str.str();

  // extract energy corrections applied 

  // ---- island in endcap

  // Get island basic clusters
  Handle<reco::BasicClusterCollection> pIslandEndcapBasicClusters;
  evt.getByLabel(islandEndcapBasicClusterProducer_, islandEndcapBasicClusterCollection_, pIslandEndcapBasicClusters);
  const reco::BasicClusterCollection* islandEndcapBasicClusters = pIslandEndcapBasicClusters.product();
  h1_nIslandEEBC_->Fill(islandEndcapBasicClusters->size());

  // fetch cluster shapes of island basic clusters in endcap
  Handle<reco::ClusterShapeCollection> pIslandEEShapes;
  evt.getByLabel(islandEndcapBasicClusterProducer_, islandEndcapBasicClusterShapes_, pIslandEEShapes);
  const reco::ClusterShapeCollection* islandEEShapes = pIslandEEShapes.product();

  // loop over the Basic clusters and fill the histogram
  iClus=0;
  for(reco::BasicClusterCollection::const_iterator aClus = islandEndcapBasicClusters->begin();
                                                    aClus != islandEndcapBasicClusters->end(); aClus++) {
    h1_islandEEBCEnergy_->Fill( aClus->energy() );
    h1_islandEEBCXtals_->Fill(  aClus->size() );
    h1_islandEEBCe5x5_->Fill( (*islandEEShapes)[iClus].e5x5() );
    h1_islandEBBCe9over25_->Fill( (*islandEEShapes)[iClus].e3x3()/(*islandEEShapes)[iClus].e5x5() );
    iClus++;
  }
  edm::LogInfo("EgammaSimpleAnalyzer") << str.str();

  // Get island super clusters
  Handle<reco::SuperClusterCollection> pIslandEndcapSuperClusters;
  evt.getByLabel(islandEndcapSuperClusterProducer_, islandEndcapSuperClusterCollection_, pIslandEndcapSuperClusters);
  const reco::SuperClusterCollection* islandEndcapSuperClusters = pIslandEndcapSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = islandEndcapSuperClusters->begin();
                                                    aClus != islandEndcapSuperClusters->end(); aClus++) {
    h1_islandEESCEnergy_->Fill( aClus->energy() );
  }


  // Get island super clusters after energy correction
  Handle<reco::SuperClusterCollection> pCorrectedIslandEndcapSuperClusters;
  evt.getByLabel(correctedIslandEndcapSuperClusterProducer_, correctedIslandEndcapSuperClusterCollection_, pCorrectedIslandEndcapSuperClusters);
  const reco::SuperClusterCollection* correctedIslandEndcapSuperClusters = pCorrectedIslandEndcapSuperClusters.product();
  h1_nIslandEESC_->Fill(islandEndcapSuperClusters->size());

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = correctedIslandEndcapSuperClusters->begin();
                                                           aClus != correctedIslandEndcapSuperClusters->end(); aClus++) {
    h1_corrIslandEESCEnergy_->Fill( aClus->energy() );
    h1_corrIslandEESCET_->Fill( aClus->energy()*sin(aClus->position().theta()) );
    h1_islandEESCClusters_->Fill( aClus->clustersSize() );
  }

  // extract energy corrections applied 


  // ----- hybrid 

  // Get hybrid super clusters
  Handle<reco::SuperClusterCollection> pHybridSuperClusters;
  evt.getByLabel(hybridSuperClusterProducer_, hybridSuperClusterCollection_, pHybridSuperClusters);
  const reco::SuperClusterCollection* hybridSuperClusters = pHybridSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = hybridSuperClusters->begin();
                                                    aClus != hybridSuperClusters->end(); aClus++) {
    h1_hybridSCEnergy_->Fill( aClus->energy() );
  }


  // Get hybrid super clusters after energy correction
  Handle<reco::SuperClusterCollection> pCorrectedHybridSuperClusters;
  evt.getByLabel(correctedHybridSuperClusterProducer_, correctedHybridSuperClusterCollection_, pCorrectedHybridSuperClusters);
  const reco::SuperClusterCollection* correctedHybridSuperClusters = pCorrectedHybridSuperClusters.product();
  h1_nHybridSC_->Fill(correctedHybridSuperClusters->size());


  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = correctedHybridSuperClusters->begin();
                                                           aClus != correctedHybridSuperClusters->end(); aClus++) {
    h1_hybridSCClusters_->Fill( aClus->clustersSize() );
    h1_corrHybridSCEnergy_->Fill( aClus->energy() );
    h1_corrHybridSCET_->Fill( aClus->energy()*sin(aClus->position().theta()) );
    h1_corrHybridSCEta_->Fill( aClus->position().eta() );
    h1_corrHybridSCPhi_->Fill( aClus->position().phi() );
  }

}

//========================================================================
void
EgammaSimpleAnalyzer::endJob() {
//========================================================================

  // go to *OUR* root file and store histograms
  rootFile_->cd();

  h1_nIslandEBBC_->Write();
  h1_nIslandEEBC_->Write();
  h1_nIslandEESC_->Write();
  h1_nHybridSC_->Write();

  h1_islandEBBCe9over25_->Write();
  h1_islandEBBCe5x5_->Write();
  h1_islandEBBCEnergy_->Write();
  h1_islandEBBCXtals_->Write();

  h1_islandEEBCe5x5_->Write();
  h1_islandEEBCEnergy_->Write();
  h1_islandEEBCXtals_->Write();

  h1_islandEESCEnergy_->Write();
  h1_corrIslandEESCEnergy_->Write();
  h1_corrIslandEESCET_->Write();
  h1_islandEESCClusters_->Write();

  h1_hybridSCClusters_->Write();
  h1_hybridSCEnergy_->Write();
  h1_corrHybridSCEnergy_->Write();
  h1_corrHybridSCET_->Write();
  h1_corrHybridSCEta_->Write();
  h1_corrHybridSCPhi_->Write();

  rootFile_->Close();
}
