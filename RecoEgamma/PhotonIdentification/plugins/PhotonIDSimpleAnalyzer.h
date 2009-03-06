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


class TFile;

//
// class declaration
//
class PhotonIDSimpleAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PhotonIDSimpleAnalyzer( const edm::ParameterSet& );
      ~PhotonIDSimpleAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
 private:

      std::string outputFile_; // output file

      // root file to store histograms
      TFile*  rootFile_;

      // data members for histograms to be filled

      // per photon quantities
      TH1F* h_isoEcalRecHit_;
      TH1F* h_isoHcalRecHit_;
      TH1F* h_trk_pt_solid_;
      TH1F* h_trk_pt_hollow_;
      TH1F* h_ntrk_solid_;
      TH1F* h_ntrk_hollow_;
      TH1F* h_r9_;
      TH1F* h_hadoverem_;
      TH1F* h_etawidth_;
      TH1F* h_ebgap_;
      TH1F* h_photonEt_;
      TH1F* h_photonEta_;
      TH1F* h_photonPhi_;
      TH1F* h_nPho_;



};
#endif
