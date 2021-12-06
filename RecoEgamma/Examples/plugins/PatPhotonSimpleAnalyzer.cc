// -*- C++ -*-
//
// Package:    PatPhotonSimpleAnalyzer
// Class:      PatPhotonSimpleAnalyzer
//
/**\class PatPhotonSimpleAnalyzer PatPhotonSimpleAnalyzer.cc RecoEgamma/PhotonIdentification/test/PatPhotonSimpleAnalyzer.cc

 Description: Generate various histograms for cuts and important
              photon ID parameters using a data sample of photons in QCD events.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M.B. Anderson
//   based on simple photon analyzer by:  J. Stilley, A. Askew
//
//         Created:  Wed Sep 23 12:00:01 CDT 2008
//

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TH1.h"
#include "TTree.h"

#include <memory>
#include <string>

class PatPhotonSimpleAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit PatPhotonSimpleAnalyzer(const edm::ParameterSet&);
  ~PatPhotonSimpleAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string outputFile_;  // output file
  double minPhotonEt_;      // minimum photon Et
  double minPhotonAbsEta_;  // min and
  double maxPhotonAbsEta_;  // max abs(eta)
  double minPhotonR9_;      // minimum R9 = E(3x3)/E(SuperCluster)
  double maxPhotonHoverE_;  // maximum HCAL / ECAL
  bool createPhotonTTree_;  // Create a TTree of photon variables

  // Will be used for creating TTree of photons.
  // These names did not have to match those from a phtn->...
  // but do match for clarity.
  struct struct_recPhoton {
    float isolationEcalRecHit;
    float isolationHcalRecHit;
    float isolationSolidTrkCone;
    float isolationHollowTrkCone;
    float nTrkSolidCone;
    float nTrkHollowCone;
    float isEBGap;
    float isEEGap;
    float isEBEEGap;
    float r9;
    float et;
    float eta;
    float phi;
    float hadronicOverEm;
    float ecalIso;
    float hcalIso;
    float trackIso;
  };
  struct_recPhoton recPhoton;

  // root file to store histograms
  TFile* rootFile_;

  // data members for histograms to be filled

  // PhotonID Histograms
  TH1F* h_isoEcalRecHit_;
  TH1F* h_isoHcalRecHit_;
  TH1F* h_trk_pt_solid_;
  TH1F* h_trk_pt_hollow_;
  TH1F* h_ntrk_solid_;
  TH1F* h_ntrk_hollow_;
  TH1F* h_ebgap_;
  TH1F* h_eeGap_;
  TH1F* h_ebeeGap_;
  TH1F* h_r9_;

  // Photon Histograms
  TH1F* h_photonEt_;
  TH1F* h_photonEta_;
  TH1F* h_photonPhi_;
  TH1F* h_hadoverem_;

  // Photon's SuperCluster Histograms
  TH1F* h_photonScEt_;
  TH1F* h_photonScEta_;
  TH1F* h_photonScPhi_;
  TH1F* h_photonScEtaWidth_;

  // Composite or Other Histograms
  TH1F* h_photonInAnyGap_;
  TH1F* h_nPassingPho_;
  TH1F* h_nPho_;

  // TTree
  TTree* tree_PhotonAll_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatPhotonSimpleAnalyzer);

using namespace std;

///////////////////////////////////////////////////////////////////////
//                           Constructor                             //
///////////////////////////////////////////////////////////////////////
PatPhotonSimpleAnalyzer::PatPhotonSimpleAnalyzer(const edm::ParameterSet& ps) {
  // Read Parameters from configuration file

  // output filename
  outputFile_ = ps.getParameter<std::string>("outputFile");
  // Read variables that must be passed to allow a
  //  supercluster to be placed in histograms as a photon.
  minPhotonEt_ = ps.getParameter<double>("minPhotonEt");
  minPhotonAbsEta_ = ps.getParameter<double>("minPhotonAbsEta");
  maxPhotonAbsEta_ = ps.getParameter<double>("maxPhotonAbsEta");
  minPhotonR9_ = ps.getParameter<double>("minPhotonR9");
  maxPhotonHoverE_ = ps.getParameter<double>("maxPhotonHoverE");

  // Read variable to that decidedes whether
  // a TTree of photons is created or not
  createPhotonTTree_ = ps.getParameter<bool>("createPhotonTTree");

  // open output file to store histograms
  rootFile_ = TFile::Open(outputFile_.c_str(), "RECREATE");
}

///////////////////////////////////////////////////////////////////////
//                            Destructor                             //
///////////////////////////////////////////////////////////////////////
PatPhotonSimpleAnalyzer::~PatPhotonSimpleAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  delete rootFile_;
}

///////////////////////////////////////////////////////////////////////
//    method called once each job just before starting event loop    //
///////////////////////////////////////////////////////////////////////
void PatPhotonSimpleAnalyzer::beginJob() {
  // go to *OUR* rootfile
  rootFile_->cd();

  // Book Histograms
  // PhotonID Histograms
  h_isoEcalRecHit_ = new TH1F("photonEcalIso", "Ecal Rec Hit Isolation", 100, 0, 100);
  h_isoHcalRecHit_ = new TH1F("photonHcalIso", "Hcal Rec Hit Isolation", 100, 0, 100);
  h_trk_pt_solid_ = new TH1F("photonTrackSolidIso", "Sum of track pT in a cone of #DeltaR", 100, 0, 100);
  h_trk_pt_hollow_ = new TH1F("photonTrackHollowIso", "Sum of track pT in a hollow cone", 100, 0, 100);
  h_ntrk_solid_ = new TH1F("photonTrackCountSolid", "Number of tracks in a cone of #DeltaR", 100, 0, 100);
  h_ntrk_hollow_ = new TH1F("photonTrackCountHollow", "Number of tracks in a hollow cone", 100, 0, 100);
  h_ebgap_ = new TH1F("photonInEBgap", "Ecal Barrel gap flag", 2, -0.5, 1.5);
  h_eeGap_ = new TH1F("photonInEEgap", "Ecal Endcap gap flag", 2, -0.5, 1.5);
  h_ebeeGap_ = new TH1F("photonInEEgap", "Ecal Barrel/Endcap gap flag", 2, -0.5, 1.5);
  h_r9_ = new TH1F("photonR9", "R9 = E(3x3) / E(SuperCluster)", 300, 0, 3);

  // Photon Histograms
  h_photonEt_ = new TH1F("photonEt", "Photon E_{T}", 200, 0, 200);
  h_photonEta_ = new TH1F("photonEta", "Photon #eta", 200, -4, 4);
  h_photonPhi_ = new TH1F("photonPhi", "Photon #phi", 200, -1. * M_PI, M_PI);
  h_hadoverem_ = new TH1F("photonHoverE", "Hadronic over EM", 200, 0, 1);

  // Photon's SuperCluster Histograms
  h_photonScEt_ = new TH1F("photonScEt", "Photon SuperCluster E_{T}", 200, 0, 200);
  h_photonScEta_ = new TH1F("photonScEta", "Photon #eta", 200, -4, 4);
  h_photonScPhi_ = new TH1F("photonScPhi", "Photon #phi", 200, -1. * M_PI, M_PI);
  h_photonScEtaWidth_ = new TH1F("photonScEtaWidth", "#eta-width", 100, 0, .1);

  // Composite or Other Histograms
  h_photonInAnyGap_ = new TH1F("photonInAnyGap", "Photon in any gap flag", 2, -0.5, 1.5);
  h_nPassingPho_ = new TH1F("photonPassingCount", "Total number photons (0=NotPassing, 1=Passing)", 2, -0.5, 1.5);
  h_nPho_ = new TH1F("photonCount", "Number of photons passing cuts in event", 10, 0, 10);

  // Create a TTree of photons if set to 'True' in config file
  if (createPhotonTTree_) {
    tree_PhotonAll_ = new TTree("TreePhotonAll", "Reconstructed Photon");
    tree_PhotonAll_->Branch(
        "recPhoton",
        &recPhoton.isolationEcalRecHit,
        "isolationEcalRecHit/"
        "F:isolationHcalRecHit:isolationSolidTrkCone:isolationHollowTrkCone:nTrkSolidCone:nTrkHollowCone:isEBGap:"
        "isEEGap:isEBEEGap:r9:et:eta:phi:hadronicOverEm:ecalIso:hcalIso:trackIso");
  }
}

///////////////////////////////////////////////////////////////////////
//                method called to for each event                    //
///////////////////////////////////////////////////////////////////////
void PatPhotonSimpleAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  using namespace std;
  using namespace edm;

  // Grab pat::Photon
  Handle<View<pat::Photon> > photonHandle;
  evt.getByLabel("selectedLayer1Photons", photonHandle);
  View<pat::Photon> photons = *photonHandle;

  int photonCounter = 0;

  for (int i = 0; i < int(photons.size()); i++) {
    pat::Photon currentPhoton = photons.at(i);

    float photonEt = currentPhoton.et();
    float superClusterEt =
        (currentPhoton.superCluster()->energy()) / (cosh(currentPhoton.superCluster()->position().eta()));

    // Only store photon candidates (SuperClusters) that pass some simple cuts
    bool passCuts = (photonEt > minPhotonEt_) && (fabs(currentPhoton.eta()) > minPhotonAbsEta_) &&
                    (fabs(currentPhoton.eta()) < maxPhotonAbsEta_) && (currentPhoton.r9() > minPhotonR9_) &&
                    (currentPhoton.hadronicOverEm() < maxPhotonHoverE_);

    if (passCuts) {
      ///////////////////////////////////////////////////////
      //                fill histograms                    //
      ///////////////////////////////////////////////////////
      // PhotonID Variables
      h_isoEcalRecHit_->Fill(currentPhoton.ecalRecHitSumEtConeDR04());
      h_isoHcalRecHit_->Fill(currentPhoton.hcalTowerSumEtConeDR04());
      h_trk_pt_solid_->Fill(currentPhoton.trkSumPtSolidConeDR04());
      h_trk_pt_hollow_->Fill(currentPhoton.trkSumPtHollowConeDR04());
      h_ntrk_solid_->Fill(currentPhoton.nTrkSolidConeDR04());
      h_ntrk_hollow_->Fill(currentPhoton.nTrkHollowConeDR04());
      h_ebgap_->Fill(currentPhoton.isEBGap());
      h_eeGap_->Fill(currentPhoton.isEEGap());
      h_ebeeGap_->Fill(currentPhoton.isEBEEGap());
      h_r9_->Fill(currentPhoton.r9());

      // Photon Variables
      h_photonEt_->Fill(photonEt);
      h_photonEta_->Fill(currentPhoton.eta());
      h_photonPhi_->Fill(currentPhoton.phi());
      h_hadoverem_->Fill(currentPhoton.hadronicOverEm());

      // Photon's SuperCluster Variables
      // eta is with respect to detector (not physics) vertex,
      // thus Et and eta are different from photon.
      h_photonScEt_->Fill(superClusterEt);
      h_photonScEta_->Fill(currentPhoton.superCluster()->position().eta());
      h_photonScPhi_->Fill(currentPhoton.superCluster()->position().phi());
      h_photonScEtaWidth_->Fill(currentPhoton.superCluster()->etaWidth());

      // It passed photon cuts, mark it
      h_nPassingPho_->Fill(1.0);

      ///////////////////////////////////////////////////////
      //                fill TTree (optional)              //
      ///////////////////////////////////////////////////////
      if (createPhotonTTree_) {
        recPhoton.isolationEcalRecHit = currentPhoton.ecalRecHitSumEtConeDR04();
        recPhoton.isolationHcalRecHit = currentPhoton.hcalTowerSumEtConeDR04();
        recPhoton.isolationSolidTrkCone = currentPhoton.trkSumPtSolidConeDR04();
        recPhoton.isolationHollowTrkCone = currentPhoton.trkSumPtHollowConeDR04();
        recPhoton.nTrkSolidCone = currentPhoton.nTrkSolidConeDR04();
        recPhoton.nTrkHollowCone = currentPhoton.nTrkHollowConeDR04();
        recPhoton.isEBGap = currentPhoton.isEBGap();
        recPhoton.isEEGap = currentPhoton.isEEGap();
        recPhoton.isEBEEGap = currentPhoton.isEBEEGap();
        recPhoton.r9 = currentPhoton.r9();
        recPhoton.et = currentPhoton.et();
        recPhoton.eta = currentPhoton.eta();
        recPhoton.phi = currentPhoton.phi();
        recPhoton.hadronicOverEm = currentPhoton.hadronicOverEm();
        recPhoton.ecalIso = currentPhoton.ecalIso();
        recPhoton.hcalIso = currentPhoton.hcalIso();
        recPhoton.trackIso = currentPhoton.trackIso();

        // Fill the tree (this records all the recPhoton.* since
        // tree_PhotonAll_ was set to point at that.
        tree_PhotonAll_->Fill();
      }

      // Record whether it was near any module gap.
      // Very convoluted at the moment.
      bool inAnyGap = currentPhoton.isEBEEGap() || (currentPhoton.isEB() && currentPhoton.isEBGap()) ||
                      (currentPhoton.isEE() && currentPhoton.isEEGap());
      if (inAnyGap) {
        h_photonInAnyGap_->Fill(1.0);
      } else {
        h_photonInAnyGap_->Fill(0.0);
      }

      photonCounter++;
    } else {
      // This didn't pass photon cuts, mark it
      h_nPassingPho_->Fill(0.0);
    }

  }  // End Loop over photons
  h_nPho_->Fill(photonCounter);
}

///////////////////////////////////////////////////////////////////////
//    method called once each job just after ending the event loop   //
///////////////////////////////////////////////////////////////////////
void PatPhotonSimpleAnalyzer::endJob() {
  // go to *OUR* root file and store histograms
  rootFile_->cd();

  // PhotonID Histograms
  h_isoEcalRecHit_->Write();
  h_isoHcalRecHit_->Write();
  h_trk_pt_solid_->Write();
  h_trk_pt_hollow_->Write();
  h_ntrk_solid_->Write();
  h_ntrk_hollow_->Write();
  h_ebgap_->Write();
  h_eeGap_->Write();
  h_ebeeGap_->Write();
  h_r9_->Write();

  // Photon Histograms
  h_photonEt_->Write();
  h_photonEta_->Write();
  h_photonPhi_->Write();
  h_hadoverem_->Write();

  // Photon's SuperCluster Histograms
  h_photonScEt_->Write();
  h_photonScEta_->Write();
  h_photonScPhi_->Write();
  h_photonScEtaWidth_->Write();

  // Composite or Other Histograms
  h_photonInAnyGap_->Write();
  h_nPassingPho_->Write();
  h_nPho_->Write();

  // Write the root file (really writes the TTree)
  rootFile_->Write();
  rootFile_->Close();
}

//define this as a plug-in
// DEFINE_FWK_MODULE(PatPhotonSimpleAnalyzer);
