// -*- C++ -*-
//
// Package:    PhotonIDSimpleAnalyzer
// Class:      PhotonIDSimpleAnalyzer
// 
/**\class PhotonIDSimpleAnalyzer PhotonIDSimpleAnalyzer.cc RecoEgamma/PhotonIdentification/test/PhotonIDSimpleAnalyzer.cc

 Description: Here I generate various histograms for cuts and important photon ID parameters using a data sample of photons in QCD events.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Stilley, A. Askew
//         Created:  Fri May 9 11:03:51 CDT 2008
// $Id$
//
///////////////////////////////////////////////////////////////////////
//                    header file for this analyzer                  //
///////////////////////////////////////////////////////////////////////
#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDSimpleAnalyzer.h"

///////////////////////////////////////////////////////////////////////
//                        CMSSW includes                             //
///////////////////////////////////////////////////////////////////////
#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
///////////////////////////////////////////////////////////////////////
//                      Root include files                           //
///////////////////////////////////////////////////////////////////////
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"

using namespace std;

///////////////////////////////////////////////////////////////////////
//                           Constructor                             //
///////////////////////////////////////////////////////////////////////
PhotonIDSimpleAnalyzer::PhotonIDSimpleAnalyzer(const edm::ParameterSet& ps)
{

// initialize output file
  outputFile_   = ps.getParameter<std::string>("outputFile");
// open output file to store histograms
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
}

///////////////////////////////////////////////////////////////////////
//                            Destructor                             //
///////////////////////////////////////////////////////////////////////
PhotonIDSimpleAnalyzer::~PhotonIDSimpleAnalyzer()
{

// do anything here that needs to be done at desctruction time
// (e.g. close files, deallocate resources etc.)

  delete rootFile_;

}


///////////////////////////////////////////////////////////////////////
//    method called once each job just before starting event loop    //
///////////////////////////////////////////////////////////////////////
void 
PhotonIDSimpleAnalyzer::beginJob(edm::EventSetup const&)
{

// go to *OUR* rootfile
  rootFile_->cd();
// book histograms
  h_isoEcalRecHit_ = new TH1F("ecalhit","Ecal Rec Hit Isolation",300,0,300);
  h_isoHcalRecHit_ = new TH1F("hcalhit","Hcal Rec Hit Isolation",300,0,300);
  h_trk_pt_solid_ = new TH1F("ptsolid", "Sum of track pT in a cone of dR" , 300, 0, 300);
  h_trk_pt_hollow_ = new TH1F("pthollow", "Sum of track pT in a hollow cone" , 300, 0, 300);
  h_ntrk_solid_ = new TH1F("ntrksol","number of tracks in a cone of dR",100,0,100);
  h_ntrk_hollow_ = new TH1F("ntrkhol","number of tracks in a hollow cone",100,0,100);
  h_r9_ = new TH1F("r9","r9 variable",300,0,3);
  h_ebgap_ = new TH1F("ebgap","EB gap flag",100,0,2);
  h_hadoverem_ = new TH1F("hadoverem","Hadronic over EM",1000,0,10);
  h_etawidth_ = new TH1F("etawidth","#eta-width",100,0,.1);
  h_nPho_ = new TH1F("nPho","Number of photons in event",10,0,10);
  h_photonEt_ = new TH1F("photonEt","Photon E_{T}",200,0,200);
  h_photonEta_ = new TH1F("photonEta","Photon #eta",400,-2,2);
  h_photonPhi_ = new TH1F("photonPhi","Photon #phi",628, -1.*TMath::Pi(), TMath::Pi());

}

///////////////////////////////////////////////////////////////////////
//                method called to for each event                    //
///////////////////////////////////////////////////////////////////////
void
PhotonIDSimpleAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es)
{
  
  using namespace std;
  using namespace edm;
  
// grab photons
  Handle<reco::PhotonCollection> photonColl;
  evt.getByLabel("photons","",photonColl);

// grab PhotonId objects  
  Handle<reco::PhotonIDAssociationCollection> photonIDMapColl;
  evt.getByLabel("PhotonIDProd","PhotonAssociatedID", photonIDMapColl);

// create reference to the object types we are interested in
  const reco::PhotonCollection *photons = photonColl.product();  
  const reco::PhotonIDAssociationCollection *phoMap = photonIDMapColl.product();

   int counter = 0;
     
   for (int i=0;i<int(photons->size());i++){   
       
     edm::Ref<reco::PhotonCollection> photonref(photonColl, i);
     reco::PhotonIDAssociationCollection::const_iterator photonIter = phoMap->find(photonref);
     const reco::PhotonIDRef &phtn = photonIter->val;
     const reco::PhotonRef &pho = photonIter->key;

     float scEt = (pho->superCluster()->energy())/(cosh(pho->superCluster()->position().eta()));
     if (scEt>20){
       h_isoEcalRecHit_->Fill((phtn)->isolationEcalRecHit());
       h_isoHcalRecHit_->Fill((phtn)->isolationHcalRecHit());
       h_trk_pt_solid_->Fill((phtn)->isolationSolidTrkCone());
       h_trk_pt_hollow_->Fill((phtn)->isolationHollowTrkCone());
       h_ntrk_solid_->Fill((phtn)->nTrkSolidCone());
       h_ntrk_hollow_->Fill((phtn)->nTrkHollowCone());
       h_ebgap_->Fill((phtn)->isEBGap());
       h_hadoverem_->Fill(pho->hadronicOverEm());
       h_etawidth_->Fill(pho->superCluster()->etaWidth());
       h_r9_->Fill((phtn)->r9());
       //Get photon object
       h_photonEt_->Fill(scEt);
       h_photonEta_->Fill(pho->superCluster()->position().eta());
       h_photonPhi_->Fill(pho->superCluster()->position().phi());

       counter++;
     }	 
   }
   h_nPho_->Fill(counter);

}

///////////////////////////////////////////////////////////////////////
//    method called once each job just after ending the event loop   //
///////////////////////////////////////////////////////////////////////
void 
PhotonIDSimpleAnalyzer::endJob()
{

// go to *OUR* root file and store histograms
  rootFile_->cd();

  h_isoEcalRecHit_->Write();
  h_isoHcalRecHit_->Write();
  h_trk_pt_solid_->Write();
  h_trk_pt_hollow_->Write();
  h_ntrk_solid_->Write();
  h_ntrk_hollow_->Write();
  h_r9_->Write();
  h_ebgap_->Write();
  h_hadoverem_->Write();
  h_etawidth_->Write();
  h_photonEt_->Write();
  h_photonEta_->Write();
  h_photonPhi_->Write();
  h_nPho_->Write();

  rootFile_->Close();

}

//define this as a plug-in
// DEFINE_FWK_MODULE(PhotonIDSimpleAnalyzer);
