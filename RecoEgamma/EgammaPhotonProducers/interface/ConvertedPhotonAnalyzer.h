#ifndef ConvertedPhotonAnalyzer_H
#define ConvertedPhotonAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <map>
#include <vector>


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TTree;
class SimVertex;
class SimTrack;

class ConvertedPhotonAnalyzer : public edm::EDAnalyzer
{

   public:
   
      //
      explicit ConvertedPhotonAnalyzer( const edm::ParameterSet& ) ;
      virtual ~ConvertedPhotonAnalyzer() {} // no need to delete ROOT stuff
                                   // as it'll be deleted upon closing TFile
      
      virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
      virtual void beginJob( const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:

  
      void fill(const std::vector<SimTrack>&, const std::vector<SimVertex>&);

     //
     std::string fOutputFileName_ ;
     TFile*      fOutputFile_ ;

     TTree *tree_;

     int nEvt_;

     std::string HepMCLabel;
     std::string SimTkLabel;
     std::string SimVtxLabel;
     std::string SimHitLabel;

     std::map<unsigned, unsigned> geantToIndex_;
     

     std::string convertedPhotonCollectionProducer_;       
     std::string convertedPhotonCollection_;
     std::string conversionTrackCandidateProducer_;
     //
     std::string outInTrackCandidateCollection_;
     std::string inOutTrackCandidateCollection_;
     //
     std::string conversionOITrackProducer_;
     std::string conversionIOTrackProducer_;

     std::string outInTrackCollection_;
     std::string inOutTrackCollection_;


     //// All MC photons
     TH1F *h_MCphoE_;
     TH1F *h_MCphoPt_;
     TH1F *h_MCphoEta_;
     //// MC Converted photons
     TH1F *h_MCConvE_;
     TH1F *h_MCConvPt_;
     TH1F *h_MCConvEta_;
     TH1F *h_MCConvR_;
     TH2F* h_MCConvRvsZ_;

     // SC from reco photons
     TH1F* h_scE_;
     TH1F* h_scEta_;
     TH1F* h_scPhi_;
     //
     TH1F* h_phoE_;
     TH1F* h_phoEta_;
     TH1F* h_phoPhi_;
     //
     // Out In Reco Tracks
     TH1F* h_OItk_inPt_;
     TH1F* h_OItk_nHits_;
     // In Out  Reco Tracks
     TH1F* h_IOtk_inPt_;
     TH1F* h_IOtk_nHits_;
     // All Reco Tracks
     TH1F* h_tk_inPt_[2];
     TH1F* h_tk_nHits_[2];
     // Reco conversion vertex position
     TH2F* h_convVtxRvsZ_;

     float mcPhoEnergy[10];
     float mcPhoEt[10];
     float mcPhoPt[10];
     float mcPhoEta[10];
     float mcPhoPhi[10];
     float mcConvR[10];
     float mcConvZ[10];
     int   idTrk1[10];
     int   idTrk2[10];

     //

     float mcPizEnergy[10];
     float mcPizEt[10];
     float mcPizPt[10];
     float mcPizEta[10];
     float mcPizPhi[10];


};

#endif
