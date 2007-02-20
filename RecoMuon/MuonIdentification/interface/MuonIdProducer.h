#ifndef MuonIdentification_MuonIdProducer_h
#define MuonIdentification_MuonIdProducer_h 1

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
// 
/*

 Description: Create a new collection of muons filling muon ID information.
              reco::TrackCollection or reco::MuonCollection can be used as input.

 Implementation:

*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonIdProducer.h,v 1.6 2007/02/20 00:35:15 dmytro Exp $
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonWithMatchInfo.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

class MuonIdProducer : public edm::EDProducer {
 public:
   enum InputMode {TrackCollection, MuonCollection};
   explicit MuonIdProducer(const edm::ParameterSet&);
   
   ~MuonIdProducer();
   
   virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
   void               fillMuonId(edm::Event&, const edm::EventSetup&,
				 reco::MuonWithMatchInfo& aMuon);
   void               init(edm::Event&, const edm::EventSetup&);
   reco::MuonWithMatchInfo*      getNewMuon(edm::Event& iEvent, 
				 const edm::EventSetup& iSetup);

   TrackDetectorAssociator trackAssociator_;
   bool useEcal_;
   bool useMuon_;
   
   bool useHcalRecHits_;
   
   edm::InputTag inputCollectionLabel_;
   std::string branchAlias_;
   InputMode mode_;

   double minPt_;
   double minP_;
   double maxAbsEta_;
   int minNumberOfMatches_;
   double maxAbsDx_;
   double maxAbsPullX_;
   double maxAbsDy_;
   double maxAbsPullY_;
   double ecalPreselectionCone_;
   double ecalSelectionCone_;
   double hcalPreselectionCone_;
   double hcalSelectionCone_;
   double muonPreselectionCone_;
   double muonSelectionCone_;
   bool debugWithTruthMatching_;
   bool inputTypeIsTrack_;

   edm::Handle<reco::TrackCollection> trackCollectionHandle_;
   reco::TrackCollection::const_iterator trackCollectionIter_;
   edm::Handle<reco::MuonCollection> muonCollectionHandle_;
   reco::MuonCollection::const_iterator muonCollectionIter_;
   int index_;
};
#endif
