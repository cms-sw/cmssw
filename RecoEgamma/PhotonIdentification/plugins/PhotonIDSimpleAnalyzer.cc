// -*- C++ -*-
//
// Package:    PhotonIDSimpleAnalyzer
// Class:      PhotonIDSimpleAnalyzer
// 
/**\class PhotonIDSimpleAnalyzer PhotonIDSimpleAnalyzer.cc RecoEgamma/PhotonIdentification/test/PhotonIDSimpleAnalyzer.cc

 Description: Here I generate various histograms for cuts and important 
              photon ID parameters using a data sample of photons in QCD events.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  J. Stilley, A. Askew
//  Editing Author:  M.B. Anderson
//
//         Created:  Fri May 9 11:03:51 CDT 2008
// $Id: PhotonIDSimpleAnalyzer.cc,v 1.3 2008/08/28 21:19:18 anderson Exp $
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
  // Read Parameters from configuration file

  // output filename
  outputFile_   = ps.getParameter<std::string>("outputFile");
  // Read variables that must be passed to allow a 
  //  supercluster to be placed in histograms as a photon.
  minPhotonEt_     = ps.getParameter<double>("minPhotonEt");
  minPhotonAbsEta_ = ps.getParameter<double>("minPhotonAbsEta");
  maxPhotonAbsEta_ = ps.getParameter<double>("maxPhotonAbsEta");
  minPhotonR9_     = ps.getParameter<double>("minPhotonR9");
  maxPhotonHoverE_ = ps.getParameter<double>("maxPhotonHoverE");

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

  // Book Histograms
  // PhotonID Histograms
  h_isoEcalRecHit_ = new TH1F("photonEcalIso",          "Ecal Rec Hit Isolation", 300, 0, 300);
  h_isoHcalRecHit_ = new TH1F("photonHcalIso",          "Hcal Rec Hit Isolation", 300, 0, 300);
  h_trk_pt_solid_  = new TH1F("photonTrackSolidIso",    "Sum of track pT in a cone of #DeltaR" , 300, 0, 300);
  h_trk_pt_hollow_ = new TH1F("photonTrackHollowIso",   "Sum of track pT in a hollow cone" ,     300, 0, 300);
  h_ntrk_solid_    = new TH1F("photonTrackCountSolid",  "Number of tracks in a cone of #DeltaR", 100, 0, 100);
  h_ntrk_hollow_   = new TH1F("photonTrackCountHollow", "Number of tracks in a hollow cone",     100, 0, 100);
  h_ebgap_         = new TH1F("photonInEBgap",          "Ecal Barrel gap flag",  2, -0.5, 1.5);
  h_eeGap_         = new TH1F("photonInEEgap",          "Ecal Endcap gap flag",  2, -0.5, 1.5);
  h_ebeeGap_       = new TH1F("photonInEEgap",          "Ecal Barrel/Endcap gap flag",  2, -0.5, 1.5);
  h_r9_            = new TH1F("photonR9",               "R9 = E(3x3) / E(SuperCluster)", 300, 0, 3);

  // Photon Histograms
  h_photonEt_      = new TH1F("photonEt",     "Photon E_{T}",  200,  0, 200);
  h_photonEta_     = new TH1F("photonEta",    "Photon #eta",   800, -4,   4);
  h_photonPhi_     = new TH1F("photonPhi",    "Photon #phi",   628, -1.*TMath::Pi(), TMath::Pi());
  h_hadoverem_     = new TH1F("photonHoverE", "Hadronic over EM", 200, 0, 1);

  // Photon's SuperCluster Histograms
  h_photonScEt_       = new TH1F("photonScEt",  "Photon SuperCluster E_{T}", 200,  0, 200);
  h_photonScEta_      = new TH1F("photonScEta", "Photon #eta",               800, -4,   4);
  h_photonScPhi_      = new TH1F("photonScPhi", "Photon #phi",628, -1.*TMath::Pi(), TMath::Pi());
  h_photonScEtaWidth_ = new TH1F("photonScEtaWidth","#eta-width",            100,  0,  .1);

  // Composite or Other Histograms
  h_photonInAnyGap_   = new TH1F("photonInAnyGap",     "Photon in any gap flag",  2, -0.5, 1.5);
  h_nPassingPho_      = new TH1F("photonPassingCount", "Total number photons (0=NotPassing, 1=Passing)", 2, -0.5, 1.5);
  h_nPho_             = new TH1F("photonCount",        "Number of photons passing cuts in event",  10,  0,  10);
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
  evt.getByLabel("photons", "", photonColl);

  // grab PhotonId objects  
  Handle<reco::PhotonIDAssociationCollection> photonIDMapColl;
  evt.getByLabel("PhotonIDProd", "PhotonAssociatedID", photonIDMapColl);

  // create reference to the object types we are interested in
  const reco::PhotonCollection *photons = photonColl.product();  
  const reco::PhotonIDAssociationCollection *phoMap = photonIDMapColl.product();

  int photonCounter = 0;
     
  for (int i=0; i<int(photons->size()); i++)
  {   
       
    edm::Ref<reco::PhotonCollection> photonref(photonColl, i);
    reco::PhotonIDAssociationCollection::const_iterator photonIter = phoMap->find(photonref);
    const reco::PhotonIDRef &phtn = photonIter->val;
    const reco::PhotonRef &pho = photonIter->key;

    float photonEt       = pho->et();
    float superClusterEt = (pho->superCluster()->energy())/(cosh(pho->superCluster()->position().eta()));

    // Only store photon candidates (SuperClusters) that pass some simple cuts
    bool passCuts = (              photonEt > minPhotonEt_     ) &&
                    (      fabs(pho->eta()) > minPhotonAbsEta_ ) &&
                    (      fabs(pho->eta()) < maxPhotonAbsEta_ ) &&
                    (          (phtn)->r9() > minPhotonR9_     ) &&
                    ( pho->hadronicOverEm() < maxPhotonHoverE_ ) ;

    if ( passCuts )
    {
      // PhotonID Variables
      h_isoEcalRecHit_->Fill((phtn)->isolationEcalRecHit());
      h_isoHcalRecHit_->Fill((phtn)->isolationHcalRecHit());
      h_trk_pt_solid_ ->Fill((phtn)->isolationSolidTrkCone());
      h_trk_pt_hollow_->Fill((phtn)->isolationHollowTrkCone());
      h_ntrk_solid_->   Fill((phtn)->nTrkSolidCone());
      h_ntrk_hollow_->  Fill((phtn)->nTrkHollowCone());
      h_ebgap_->        Fill((phtn)->isEBGap());
      h_eeGap_->        Fill((phtn)->isEEGap()); 
      h_ebeeGap_->      Fill((phtn)->isEBEEGap());
      h_r9_->           Fill((phtn)->r9());

      // Photon Variables
      h_photonEt_->  Fill(photonEt);
      h_photonEta_-> Fill(pho->eta());
      h_photonPhi_-> Fill(pho->phi());
      h_hadoverem_-> Fill(pho->hadronicOverEm());

      // Photon's SuperCluster Variables
      // eta is with respect to detector (not physics) vertex,
      // thus Et and eta are different from photon.
      h_photonScEt_->      Fill(superClusterEt);
      h_photonScEta_->     Fill(pho->superCluster()->position().eta());
      h_photonScPhi_->     Fill(pho->superCluster()->position().phi());
      h_photonScEtaWidth_->Fill(pho->superCluster()->etaWidth());

      // It passed photon cuts, mark it
      h_nPassingPho_->Fill(1.0);

      // Record whether it was near any module gap.
      // Very convoluted at the moment.
      bool inAnyGap = (phtn)->isEBEEGap() || ((phtn)->isEBPho()&&(phtn)->isEBGap()) || ((phtn)->isEEPho()&&(phtn)->isEEGap());
      if (inAnyGap) {
        h_photonInAnyGap_->Fill(1.0);
      } else {
        h_photonInAnyGap_->Fill(0.0);
      }

      photonCounter++;
    } 
    else
    {
      // This didn't pass photon cuts, mark it
      h_nPassingPho_->Fill(0.0);
    }

  } // End Loop over photons
  h_nPho_->Fill(photonCounter);

}

///////////////////////////////////////////////////////////////////////
//    method called once each job just after ending the event loop   //
///////////////////////////////////////////////////////////////////////
void 
PhotonIDSimpleAnalyzer::endJob()
{

  // go to *OUR* root file and store histograms
  rootFile_->cd();

  // PhotonID Histograms
  h_isoEcalRecHit_->Write();
  h_isoHcalRecHit_->Write();
  h_trk_pt_solid_-> Write();
  h_trk_pt_hollow_->Write();
  h_ntrk_solid_->   Write();
  h_ntrk_hollow_->  Write();
  h_ebgap_->     Write();
  h_eeGap_->     Write();
  h_ebeeGap_->   Write();
  h_r9_->        Write();

  // Photon Histograms
  h_photonEt_->  Write();
  h_photonEta_-> Write();
  h_photonPhi_-> Write();
  h_hadoverem_-> Write();

  // Photon's SuperCluster Histograms
  h_photonScEt_->      Write();
  h_photonScEta_->     Write();
  h_photonScPhi_->     Write();
  h_photonScEtaWidth_->Write();

  // Composite or Other Histograms
  h_photonInAnyGap_->Write();
  h_nPassingPho_->   Write();
  h_nPho_->          Write();

  rootFile_->Close();

}

//define this as a plug-in
// DEFINE_FWK_MODULE(PhotonIDSimpleAnalyzer);
