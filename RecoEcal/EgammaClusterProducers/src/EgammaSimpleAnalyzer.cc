/**\class EgammaSimpleAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Shahram Rahatlou
//         Created:  10 May 2006
// $Id: EgammaSimpleAnalyzer.cc,v 1.4 2006/06/16 10:16:43 rahatlou Exp $
//

#include "RecoEcal/EgammaClusterProducers/interface/EgammaSimpleAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Handle.h"

#include "TFile.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


//========================================================================
EgammaSimpleAnalyzer::EgammaSimpleAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{

  xMinHist_ = ps.getParameter<double>("xMinHist");
  xMaxHist_ = ps.getParameter<double>("xMaxHist");
  nbinHist_ = ps.getParameter<int>("nbinHist");

  islandBasicClusterCollection_ = ps.getParameter<std::string>("islandBasicClusterCollection");
  islandBasicClusterProducer_   = ps.getParameter<std::string>("islandBasicClusterProducer");

  islandSuperClusterCollection_ = ps.getParameter<std::string>("islandSuperClusterCollection");
  islandSuperClusterProducer_   = ps.getParameter<std::string>("islandSuperClusterProducer");

  correctedIslandSuperClusterCollection_ = ps.getParameter<std::string>("correctedIslandSuperClusterCollection");
  correctedIslandSuperClusterProducer_   = ps.getParameter<std::string>("correctedIslandSuperClusterProducer");

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
EgammaSimpleAnalyzer::beginJob(edm::EventSetup const&) {
//========================================================================

  // go to *OUR* rootfile and book histograms
  rootFile_->cd();
  h1_islandBCEnergy_ = new TH1F("islandBCEnergy","Energy of basic clusters with island algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_islandSCEnergy_ = new TH1F("islandSCEnergy","Energy of super clusters with island algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrIslandSCEnergy_ = new TH1F("corrIslandSCEnergy","Corrected Energy of super clusters with island algo",nbinHist_,xMinHist_,xMaxHist_);

  h1_hybridSCEnergy_ = new TH1F("hybridSCEnergy","Energy of super clusters with hybrid algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrHybridSCEnergy_ = new TH1F("corrHybridSCEnergy","Corrected Energy of super clusters with hybrid algo",nbinHist_,xMinHist_,xMaxHist_);
  h1_corrHybridSCEta_ = new TH1F("corrHybridSCEta","Eta of super clusters with hybrid algo",40,-3.,3.);
  h1_corrHybridSCPhi_ = new TH1F("corrHybridSCPhi","Phi of super clusters with hybrid algo",40,0.,6.28);

}


//========================================================================
void
EgammaSimpleAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classes

  // Get island basic clusters
  Handle<reco::BasicClusterCollection> pIslandBasicClusters;
  try {
    evt.getByLabel(islandBasicClusterProducer_, islandBasicClusterCollection_, pIslandBasicClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << islandBasicClusterCollection_.c_str() ;
  }
  const reco::BasicClusterCollection* islandBasicClusters = pIslandBasicClusters.product();

  // loop over the Basic clusters and fill the histogram
  for(reco::BasicClusterCollection::const_iterator aClus = islandBasicClusters->begin();
                                                    aClus != islandBasicClusters->end(); aClus++) {
    h1_islandBCEnergy_->Fill( aClus->energy()*sin(aClus->position().theta()) );
  }

  // Get island super clusters
  Handle<reco::SuperClusterCollection> pIslandSuperClusters;
  try {
    evt.getByLabel(islandSuperClusterProducer_, islandSuperClusterCollection_, pIslandSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << islandSuperClusterCollection_.c_str() ;
  }
  const reco::SuperClusterCollection* islandSuperClusters = pIslandSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = islandSuperClusters->begin();
                                                    aClus != islandSuperClusters->end(); aClus++) {
    h1_islandSCEnergy_->Fill( aClus->energy()*sin(aClus->position().theta()) );
  }


  // Get island super clusters after energy correction
  Handle<reco::SuperClusterCollection> pCorrectedIslandSuperClusters;
  try {
    evt.getByLabel(correctedIslandSuperClusterProducer_, correctedIslandSuperClusterCollection_, pCorrectedIslandSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << correctedIslandSuperClusterCollection_.c_str() ;
  }
  const reco::SuperClusterCollection* correctedIslandSuperClusters = pCorrectedIslandSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = correctedIslandSuperClusters->begin();
                                                           aClus != correctedIslandSuperClusters->end(); aClus++) {
    h1_corrIslandSCEnergy_->Fill( aClus->energy()*sin(aClus->position().theta()) );
  }



  // Get hybrid super clusters
  Handle<reco::SuperClusterCollection> pHybridSuperClusters;
  try {
    evt.getByLabel(hybridSuperClusterProducer_, hybridSuperClusterCollection_, pHybridSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << hybridSuperClusterCollection_.c_str() ;
  }
  const reco::SuperClusterCollection* hybridSuperClusters = pHybridSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = hybridSuperClusters->begin();
                                                    aClus != hybridSuperClusters->end(); aClus++) {
    h1_hybridSCEnergy_->Fill( aClus->energy()*sin(aClus->position().theta()) );
  }


  // Get hybrid super clusters after energy correction
  Handle<reco::SuperClusterCollection> pCorrectedHybridSuperClusters;
  try {
    evt.getByLabel(correctedHybridSuperClusterProducer_, correctedHybridSuperClusterCollection_, pCorrectedHybridSuperClusters);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << correctedHybridSuperClusterCollection_.c_str() ;
  }
  const reco::SuperClusterCollection* correctedHybridSuperClusters = pCorrectedHybridSuperClusters.product();

  // loop over the super clusters and fill the histogram
  for(reco::SuperClusterCollection::const_iterator aClus = correctedHybridSuperClusters->begin();
                                                           aClus != correctedHybridSuperClusters->end(); aClus++) {
    h1_corrHybridSCEnergy_->Fill( aClus->energy()*sin(aClus->position().theta()) );
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

  h1_islandBCEnergy_->Write();
  h1_islandSCEnergy_->Write();
  h1_corrIslandSCEnergy_->Write();

  h1_hybridSCEnergy_->Write();
  h1_corrHybridSCEnergy_->Write();
  h1_corrHybridSCEta_->Write();
  h1_corrHybridSCPhi_->Write();

  rootFile_->Close();
}
