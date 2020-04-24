#ifndef RecoEgamma_PhotonIdentification_PhotonIDSimpleAnalyzer_H
#define RecoEgamma_PhotonIdentification_PhotonIDSimpleAnalyzer_H

/**\class PhotonIDSimpleAnalyzer

 Description: Analyzer to make a load of histograms for the improvement of the PhotonID object

 Implementation:
     \\\author: J. Stilley, A. Askew May 2008
*/
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

#include <string>
#include "TH1.h"
#include "TTree.h"


class TFile;

//
// class declaration
//
class PhotonIDSimpleAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PhotonIDSimpleAnalyzer( const edm::ParameterSet& );
      ~PhotonIDSimpleAnalyzer() override;


      void analyze( const edm::Event&, const edm::EventSetup& ) override;
      void beginJob() override;
      void endJob() override;
 private:

      std::string outputFile_;   // output file
      double minPhotonEt_;       // minimum photon Et
      double minPhotonAbsEta_;   // min and
      double maxPhotonAbsEta_;   // max abs(eta)
      double minPhotonR9_;       // minimum R9 = E(3x3)/E(SuperCluster)
      double maxPhotonHoverE_;   // maximum HCAL / ECAL 
      bool   createPhotonTTree_; // Create a TTree of photon variables

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
        float isEBEtaGap;
        float isEBPhiGap;
	float isEERingGap;
	float isEEDeeGap;
	float isEBEEGap;
	float r9;
	float et;
	float eta;
	float phi;
        float hadronicOverEm;
      } ;
      struct_recPhoton recPhoton;

      // root file to store histograms
      TFile*  rootFile_;

      // data members for histograms to be filled

      // PhotonID Histograms
      TH1F* h_isoEcalRecHit_;
      TH1F* h_isoHcalRecHit_;
      TH1F* h_trk_pt_solid_;
      TH1F* h_trk_pt_hollow_;
      TH1F* h_ntrk_solid_;
      TH1F* h_ntrk_hollow_;
      TH1F* h_ebetagap_;
      TH1F* h_ebphigap_;
      TH1F* h_eeringGap_;
      TH1F* h_eedeeGap_;
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
      TH1F* h_nPassEM_;
      TH1F* h_nPho_;

      // TTree
      TTree* tree_PhotonAll_;
};
#endif
