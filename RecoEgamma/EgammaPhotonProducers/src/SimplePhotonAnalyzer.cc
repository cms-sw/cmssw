/**\class PhotonSimpleAnalyzer
 **
 ** $Date: $ 
 ** $Revision: $
 ** \author Nancy Marinelli, U. of Notre Dame, US
*/

#include "RecoEgamma/EgammaPhotonProducers/interface/SimplePhotonAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Handle.h"

#include "TFile.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//========================================================================
SimplePhotonAnalyzer::SimplePhotonAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{

  xMinHist_ = ps.getParameter<double>("xMinHist");
  xMaxHist_ = ps.getParameter<double>("xMaxHist");
  nbinHist_ = ps.getParameter<int>("nbinHist");

  photonCollectionProducer_ = ps.getParameter<std::string>("phoProducer");
  photonCorrCollectionProducer_ = ps.getParameter<std::string>("corrPhoProducer");
  uncorrectedPhotonCollection_ = ps.getParameter<std::string>("uncorrectedPhotonCollection");
  correctedPhotonCollection_ = ps.getParameter<std::string>("correctedPhotonCollection");
 


  outputFile_   = ps.getParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms

}


//========================================================================
SimplePhotonAnalyzer::~SimplePhotonAnalyzer()
//========================================================================
{

  delete rootFile_;
}

//========================================================================
void
SimplePhotonAnalyzer::beginJob(edm::EventSetup const&) {
//========================================================================

  // go to *OUR* rootfile and book histograms
  rootFile_->cd();

  h1_scE_ = new TH1F("scE","Uncorrected photons : Island SC Energy ",100,0., 50.);
  h1_scEta_ = new TH1F("scEta","Uncorrected photons: Island SC Eta ",40,-3., 3.);
  h1_scPhi_ = new TH1F("scPhi","Uncorrected photons: Island SC Phi ",40,0., 6.28);

  h1_corrPho_scE_ = new TH1F("scCorrE","Corrected photons : Island SC Energy ",100,0., 50.);
  h1_corrPho_scEta_ = new TH1F("scCorrEta","Corrected photons: Island SC Eta ",40,-3., 3.);
  h1_corrPho_scPhi_ = new TH1F("scCorrPhi","Corrected photons: Island SC Phi ",40,0., 6.28);


}


//========================================================================
void
SimplePhotonAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
//========================================================================

  using namespace edm; // needed for all fwk related classes
  edm::LogInfo("PhotonAnalyzer") << "Analyzing event number: " << evt.id() << "\n";

  //  ----- barrel with island

  // Get theuncorrected  photon collection
 Handle<reco::PhotonCollection> uncorrectedPhotonHandle; 
  try {
    evt.getByLabel(photonCollectionProducer_, uncorrectedPhotonCollection_ , uncorrectedPhotonHandle);
  } catch ( cms::Exception& ex ) {
    edm::LogError("SimplePhotonAnalyzer") << "Error! can't get collection with label " << uncorrectedPhotonCollection_.c_str() ;
  }
  const reco::PhotonCollection phoCollection = *(uncorrectedPhotonHandle.product());




  // Get the  corrected  photon collection
 Handle<reco::PhotonCollection> correctedPhotonHandle; 
  try {
    evt.getByLabel(photonCorrCollectionProducer_, correctedPhotonCollection_ , correctedPhotonHandle);
  } catch ( cms::Exception& ex ) {
    edm::LogError("SimplePhotonAnalyzer") << "Error! can't get collection with label " << correctedPhotonCollection_.c_str() ;
  }
  const reco::PhotonCollection corrPhoCollection = *(correctedPhotonHandle.product());






  // Loop over uncorrected  Photon candidates 
  for( reco::PhotonCollection::const_iterator  iPho = phoCollection.begin(); iPho != phoCollection.end(); iPho++) {
    
    /////  Fill histos
   

    h1_scE_->Fill( (*iPho).superCluster()->energy() );
    h1_scEta_->Fill( (*iPho).superCluster()->position().eta() );
    h1_scPhi_->Fill( (*iPho).superCluster()->position().phi() );



    
  }  



  // Loop over corrected  Photon candidates 
  for( reco::PhotonCollection::const_iterator  iPho = corrPhoCollection.begin(); iPho != corrPhoCollection.end(); iPho++) {
    
    /////  Fill histos
   

    h1_corrPho_scE_->Fill( (*iPho).superCluster()->energy() );
    h1_corrPho_scEta_->Fill( (*iPho).superCluster()->position().eta() );
    h1_corrPho_scPhi_->Fill( (*iPho).superCluster()->position().phi() );
    
  }  





}

//========================================================================
void
SimplePhotonAnalyzer::endJob() {
//========================================================================


  rootFile_->cd();

  h1_scE_->Write();
  h1_scEta_->Write();
  h1_scPhi_->Write();


  h1_corrPho_scE_->Write();
  h1_corrPho_scEta_->Write();
  h1_corrPho_scPhi_->Write();


  rootFile_->Close();
}
