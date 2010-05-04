#ifndef RecoMET_MuonTCMETValueMapProducer_h
#define RecoMET_MuonTCMETValueMapProducer_h
// -*- C++ -*-
//
// Package:    Test
// Class:      Test
// 
/**\class MuonTCMETValueMapProducer MuonTCMETValueMapProducer.cc JetMETCorrections/MuonTCMETValueMapProducer/src/MuonTCMETValueMapProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Created:  Wed Aug 29 2007
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TH2.h"
#include "TVector3.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace cms {
class MuonTCMETValueMapProducer : public edm::EDProducer {
   public:
      explicit MuonTCMETValueMapProducer(const edm::ParameterSet&);
      ~MuonTCMETValueMapProducer();

      TH2D* getResponseFunction_fit ( );
      TH2D* getResponseFunction_mode ( );

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      //list of cuts
      edm::Handle<reco::MuonCollection>  muon_h;
      edm::Handle<reco::BeamSpot>        beamSpot_h;

      edm::InputTag muonInputTag_;  
      edm::InputTag beamSpotInputTag_;

      const class MagneticField* bField;

      class TH2D* response_function;

      bool muonGlobal_;
      bool muonTracker_;
      bool useCaloMuons_;

      int rfType_;

      double  minpt_;
      double  maxpt_;
      double  maxeta_;
      double  maxchi2_;
      double  minhits_;
      double  maxd0_;
      double  maxPtErr_;
      std::vector<int> trkQuality_;
      std::vector<int> trkAlgos_;

      int     muonMinValidStaHits_;
      double  muonpt_;
      double  muoneta_;
      double  muonchi2_;
      double  muonhits_;
      double  muond0_;
      double  muonDeltaR_;

      //functions
      bool isGoodMuon( const reco::Muon* );
      bool isGoodCaloMuon( const reco::Muon*, const unsigned int );
      bool isGoodTrack( const reco::Muon* );
      class TVector3 propagateTrack( const reco::Muon* );
  };
}
#endif


