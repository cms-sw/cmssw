// -*- C++ -*-
//
// Package:    SimplePi0DiscAnalyzer
// Class:      SimplePi0DiscAnalyzer
// 
//
// Original Authors:  A. Kyriakis NCSR "Demokritos" Athens
//                    D Maletic, "Vinca" Belgrade
//
//         Created:  Wed Sep 12 13:36:27 CEST 2007
// $Id: SimplePi0DiscAnalyzer.cc,v 1.4 2007/12/08 10:59:22 nancy Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"

#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace reco;
using namespace edm;

//
// class decleration
//

class SimplePi0DiscAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SimplePi0DiscAnalyzer(const edm::ParameterSet&);

      ~SimplePi0DiscAnalyzer();


   private:
     
     virtual void beginJob(const edm::EventSetup&) ;
     virtual void analyze(const edm::Event&, const edm::EventSetup&);
     virtual void endJob() ;
      // ----------member data ---------------------------
	
     string photonCollectionProducer_;       
     string photonCorrCollectionProducer_;       
     string uncorrectedPhotonCollection_;
     string correctedPhotonCollection_;

     string outputFile_;
     TFile*  rootFile_;

     TH1F* hConv_ntracks_;

     TH1F* hAll_nnout_Assoc_;
     TH1F* hAll_nnout_NoConv_Assoc_;
     TH1F* hBarrel_nnout_Assoc_;
     TH1F* hBarrel_nnout_NoConv_Assoc_;
     TH1F* hEndcNoPresh_nnout_Assoc_;
     TH1F* hEndcNoPresh_nnout_NoConv_Assoc_;
     TH1F* hEndcWithPresh_nnout_Assoc_;
     TH1F* hEndcWithPresh_nnout_NoConv_Assoc_;
     
};

SimplePi0DiscAnalyzer::SimplePi0DiscAnalyzer(const edm::ParameterSet& iConfig)

{

  photonCollectionProducer_ = iConfig.getParameter<string>("phoProducer");
  photonCorrCollectionProducer_ = iConfig.getParameter<string>("corrPhoProducer");
  uncorrectedPhotonCollection_ = iConfig.getParameter<string>("uncorrectedPhotonCollection");
  correctedPhotonCollection_ = iConfig.getParameter<string>("correctedPhotonCollection");

    outputFile_   = iConfig.getParameter<string>("outputFile");

    rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms

}

SimplePi0DiscAnalyzer::~SimplePi0DiscAnalyzer()
{
   delete rootFile_;

}

//
// member functions
//

// ------------ method called to for each event  ------------
void
SimplePi0DiscAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << endl;
  cout << " -------------- NEW EVENT : Run, Event =  " << iEvent.id() << endl;
   
  // Get the ConvertedPhotonCollection
  Handle<ConversionCollection> convertedPhotonHandle; // get the Converted Photon info
  iEvent.getByLabel("conversions", "", convertedPhotonHandle);
  const reco::ConversionCollection phoCollection = *(convertedPhotonHandle.product());

  cout << " ---> ConvertedPhotonCollection.size() = " << phoCollection.size() << endl;

  // Get the  corrected  photon collection
  Handle<reco::PhotonCollection> correctedPhotonHandle; 
  iEvent.getByLabel(photonCorrCollectionProducer_, correctedPhotonCollection_ , correctedPhotonHandle);
  const reco::PhotonCollection photons = *(correctedPhotonHandle.product());

  cout <<"----> Photons size: "<< photons.size()<<endl;

  edm::Handle<reco::PhotonPi0DiscriminatorAssociationMap>  map;
  iEvent.getByLabel("piZeroDiscriminators","PhotonPi0DiscriminatorAssociationMap",  map);
  reco::PhotonPi0DiscriminatorAssociationMap::const_iterator mapIter;

  int PhoInd = 0;

  /*
      
  for( reco::PhotonCollection::const_iterator  iPho = photons.begin(); iPho != photons.end(); iPho++) { // Loop over Photons
          
    reco::Photon localPho(*iPho);

    float Photon_eta = localPho.eta(); float Photon_phi = localPho.phi();
    cout << "Photon Id = " << PhoInd << " with Energy = " << localPho.energy() 
         << " Et = " <<  localPho.energy()*sin(2*atan(exp(-Photon_eta)))
         << " Eta = " << Photon_eta << " and Phi = " << Photon_phi << endl;       

    SuperClusterRef it_super = localPho.superCluster(); // get the SC related to the Photon candidate

    bool isPhotConv = false;
    int Ntrk_conv = 0;
    int Conv_SCE_id = 0;
    for( reco::ConversionCollection::const_iterator iCPho = phoCollection.begin(); 
	 iCPho != phoCollection.end(); iCPho++) { 
       SuperClusterRef it_superConv = (*iCPho).superCluster();// get the SC related to the  Converted Photon candidate
       if(it_super == it_superConv) { 
         isPhotConv = (*iCPho).isConverted(); 
         Ntrk_conv = (*iCPho).tracks().size();
	 break;
       }    
       Conv_SCE_id++;
    } // End of Photon Conversion Loop     
    hConv_ntracks_->Fill(Ntrk_conv);
    
    mapIter = map->find(edm::Ref<reco::PhotonCollection>(correctedPhotonHandle,PhoInd));
    float nn = mapIter->val;
    if(fabs(Photon_eta) <= 1.442) {
       hBarrel_nnout_Assoc_->Fill(nn);
       hAll_nnout_Assoc_->Fill(nn);
       cout << "AssociationMap Barrel NN = " << nn << endl;
       if(isPhotConv) {
	  hBarrel_nnout_NoConv_Assoc_->Fill(nn);
	  hAll_nnout_NoConv_Assoc_->Fill(nn);          
       } 
    } else if( (fabs(Photon_eta) >= 1.556 && fabs(Photon_eta) < 1.65) || fabs(Photon_eta) > 2.5) {     
       hEndcNoPresh_nnout_Assoc_->Fill(nn);
       hAll_nnout_Assoc_->Fill(nn);
       cout << "AssociationMap EndcNoPresh NN = " << nn << endl;
       if(isPhotConv) {
	  hEndcNoPresh_nnout_NoConv_Assoc_->Fill(nn);
	  hAll_nnout_NoConv_Assoc_->Fill(nn);
       }
    } else if(fabs(Photon_eta) >= 1.65 && fabs(Photon_eta) <= 2.5 ) {
       hEndcWithPresh_nnout_Assoc_->Fill(nn);
       hAll_nnout_Assoc_->Fill(nn);
       cout << "AssociationMap EndcWithPresh NN = " << nn << endl;
       if(isPhotConv) {
	  hEndcWithPresh_nnout_NoConv_Assoc_->Fill(nn);
	  hAll_nnout_NoConv_Assoc_->Fill(nn);
       }
    } 

    PhoInd++;
  } // End Loop over Photons



  */

}

// ------------ method called once each job just before starting event loop  ------------
void SimplePi0DiscAnalyzer::beginJob(const edm::EventSetup&)
{
  rootFile_->cd();

  hConv_ntracks_ = new TH1F("nConvTracks","Number of tracks of converted Photons ",10,0.,10);    
  hAll_nnout_Assoc_ = new TH1F("All_nnout_Assoc","NNout for All Photons(AssociationMap)",100,0.,1.);
  hAll_nnout_NoConv_Assoc_ = new TH1F("All_nnout_NoConv_Assoc","NNout for Unconverted Photons(AssociationMap)",100,0.,1.);
  hBarrel_nnout_Assoc_ = new TH1F("barrel_nnout_Assoc","NNout for Barrel Photons(AssociationMap)",100,0.,1.);
  hBarrel_nnout_NoConv_Assoc_ = new TH1F("barrel_nnout_NoConv_Assoc","NNout for Barrel Unconverted Photons(AssociationMap)",100,0.,1.);
  hEndcNoPresh_nnout_Assoc_ = new TH1F("endcNoPresh_nnout_Assoc","NNout for Endcap NoPresh Photons(AssociationMap)",100,0.,1.);
  hEndcNoPresh_nnout_NoConv_Assoc_ = new TH1F("endcNoPresh_nnout_NoConv_Assoc","NNout for Endcap Unconverted NoPresh Photons(AssociationMap)",100,0.,1.);
  hEndcWithPresh_nnout_Assoc_ = new TH1F("endcWithPresh_nnout_Assoc","NNout for Endcap WithPresh Photons(AssociationMap)",100,0.,1.);
  hEndcWithPresh_nnout_NoConv_Assoc_ = new TH1F("endcWithPresh_nnout_NoConv_Assoc","NNout for Endcap Unconverted WithPresh Photons(AssociationMap)",100,0.,1.);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimplePi0DiscAnalyzer::endJob() {
  rootFile_->cd();

  hConv_ntracks_->Write();

  hAll_nnout_Assoc_->Write();
  hAll_nnout_NoConv_Assoc_->Write();
  hBarrel_nnout_Assoc_->Write();
  hBarrel_nnout_NoConv_Assoc_->Write();
  hEndcNoPresh_nnout_Assoc_->Write();
  hEndcNoPresh_nnout_NoConv_Assoc_->Write();
  hEndcWithPresh_nnout_Assoc_->Write();
  hEndcWithPresh_nnout_NoConv_Assoc_->Write();

  rootFile_->Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(SimplePi0DiscAnalyzer);
