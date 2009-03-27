#ifndef Type1MET_MuonTCMETValueMapProducer_h
#define Type1MET_MuonTCMETValueMapProducer_h
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
      TH2D* getResponseFunction ( );

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

      double  minpt_;
      double  maxpt_;
      double  maxeta_;
      double  maxchi2_;
      double  minhits_;
      double  maxd0_;

      double  muonpt_;
      double  muoneta_;
      double  muonchi2_;
      double  muonhits_;
      double  muond0_;
      bool    muonGlobal_ ;
      bool    muonTracker_;
      double  muonQoverPerror_;
      double  muondeltaPt_;

      //functions
      bool isGoodMuon( const reco::Muon* );
      bool isGoodGlobalMuon( const reco::Muon* );
      bool isGoodTrack( const reco::Muon* );
      class TVector3 propagateTrack( const reco::Muon* );
  };
}
#endif


