#ifndef RecoMET_MuonMETValueMapProducer_h
#define RecoMET_MuonMETValueMapProducer_h
// -*- C++ -*-
//
// Package:    Test
// Class:      Test
// 
/**\class MuonMETValueMapProducer MuonMETValueMapProducer.cc JetMETCorrections/MuonMETValueMapProducer/src/MuonMETValueMapProducer.cc

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
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

namespace cms {
class MuonMETValueMapProducer : public edm::EDProducer {
   public:
      explicit MuonMETValueMapProducer(const edm::ParameterSet&);
      ~MuonMETValueMapProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      //list of cuts
      edm::InputTag beamSpotInputTag_;
      edm::InputTag muonInputTag_;
      bool useTrackAssociatorPositions_;
      bool useRecHits_;
      bool useHO_;
      bool isAlsoTkMu_;
      double towerEtThreshold_;
      double minPt_;
      double maxEta_;
      double maxNormChi2_;
      double maxd0_;
      int minnHits_;
      int minnValidStaHits_;
      TrackDetectorAssociator   trackAssociator_;
      TrackAssociatorParameters trackAssociatorParameters_;
  };
}
#endif


