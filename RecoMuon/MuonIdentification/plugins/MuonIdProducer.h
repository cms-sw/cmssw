#ifndef MuonIdentification_MuonIdProducer_h
#define MuonIdentification_MuonIdProducer_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
// 
/*

 Description: reco::Muon producer that can fill various information:
              - track-segment matching
              - energy deposition
              - muon isolation
              - muon hypothesis compatibility (calorimeter)
              Acceptable inputs:
              - reco::TrackCollection
              - reco::MuonCollection
              - reco::MuonTrackLinksCollection
*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonIdProducer.h,v 1.12 2007/05/01 18:18:29 dmytro Exp $
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"

class MuonIdProducer : public edm::EDProducer {
 public:
   enum InputMode {TrackCollection=0, MuonCollection=1, LinkCollection=2};
   explicit MuonIdProducer(const edm::ParameterSet&);
   
   virtual ~MuonIdProducer();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   void          fillMuonId( edm::Event&, const edm::EventSetup&, reco::Muon& aMuon );
   void          fillArbitrationInfo( reco::MuonCollection* );
   void          fillMuonIsolation( edm::Event&, const edm::EventSetup&, reco::Muon& aMuon );
   void          init( edm::Event&, const edm::EventSetup& );
   reco::Muon*   nextMuon( edm::Event& iEvent, const edm::EventSetup& iSetup );
   reco::Muon*   makeMuon( const reco::Track& track );
   
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters parameters_;
   
   edm::InputTag inputTrackCollectionLabel_;
   edm::InputTag inputMuonCollectionLabel_;
   edm::InputTag inputLinkCollectionLabel_;
   std::string branchAlias_;
   InputMode mode_;

   // selections
   double minPt_;
   double minP_;
   int    minNumberOfMatches_;
   double stiffMinPt_;
   double stiffMinP_;
   int    stiffMinNumberOfMatches_;
   double maxAbsEta_;
   
   // matching
   double maxAbsDx_;
   double maxAbsPullX_;
   double maxAbsDy_;
   double maxAbsPullY_;
   
   // what information to fill
   bool fillCaloCompatibility_;
   bool fillEnergy_;
   bool fillMatching_;
   bool fillIsolation_;
   
   bool debugWithTruthMatching_;

   edm::Handle<reco::TrackCollection>             trackCollectionHandle_;
   edm::Handle<reco::MuonCollection>              muonCollectionHandle_;
   edm::Handle<reco::MuonTrackLinksCollection>    linkCollectionHandle_;
   reco::TrackCollection::const_iterator          trackCollectionIter_;
   reco::MuonCollection::const_iterator           muonCollectionIter_;
   reco::MuonTrackLinksCollection::const_iterator linkCollectionIter_;
   int index_;
   
   MuonCaloCompatibility muonCaloCompatibility_;
};
#endif
