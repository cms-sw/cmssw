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
// $Id: MuonIdProducer.h,v 1.29 2013/02/25 21:29:14 chrjones Exp $
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
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
// #include "Utilities/Timing/interface/TimerStack.h"

#include "RecoMuon/MuonIdentification/interface/MuonTimingFiller.h"
#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

class MuonMesh;
class MuonKinkFinder;

class MuonIdProducer : public edm::EDProducer {
 public:
   typedef reco::Muon::MuonTrackType TrackType;
  
   explicit MuonIdProducer(const edm::ParameterSet&);
   
   virtual ~MuonIdProducer();
   
   virtual void produce(edm::Event&, const edm::EventSetup&) override;
   virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
   
   static double sectorPhi( const DetId& id );

 private:
   void          fillMuonId( edm::Event&, const edm::EventSetup&, reco::Muon&, 
			     TrackDetectorAssociator::Direction direction = TrackDetectorAssociator::InsideOut );
   void          fillArbitrationInfo( reco::MuonCollection* );
   void          fillMuonIsolation( edm::Event&, const edm::EventSetup&, reco::Muon& aMuon,
				    reco::IsoDeposit& trackDep, reco::IsoDeposit& ecalDep, reco::IsoDeposit& hcalDep, reco::IsoDeposit& hoDep,
				    reco::IsoDeposit& jetDep);
   void          fillGlbQuality( edm::Event&, const edm::EventSetup&, reco::Muon& aMuon );
   void          fillTrackerKink( reco::Muon& aMuon ); 
   void          init( edm::Event&, const edm::EventSetup& );
   
   // make a muon based on a track ref
   reco::Muon    makeMuon( edm::Event& iEvent, const edm::EventSetup& iSetup, 
			   const reco::TrackRef& track, TrackType type);
   // make a global muon based on the links object
   reco::Muon    makeMuon( const reco::MuonTrackLinks& links );
   
   // make a muon based on track (p4)
   reco::Muon    makeMuon( const reco::Track& track );
   
   reco::CaloMuon makeCaloMuon( const reco::Muon& );

   // check if a silicon track satisfies the trackerMuon requirements
   bool          isGoodTrack( const reco::Track& track );
   
   bool          isGoodTrackerMuon( const reco::Muon& muon );
   bool          isGoodRPCMuon( const reco::Muon& muon );
   
   // check number of common DetIds for a given trackerMuon and a stand alone
   // muon track
   int           overlap(const reco::Muon& muon, const reco::Track& track);

   unsigned int  chamberId(const DetId&);
   
   double phiOfMuonIneteractionRegion( const reco::Muon& muon ) const;

   bool checkLinks(const reco::MuonTrackLinks*) const ;
     
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters parameters_;
   
   std::vector<edm::InputTag> inputCollectionLabels_;
   std::vector<std::string>   inputCollectionTypes_;

   MuonTimingFiller* theTimingFiller_;

   // selections
   double minPt_;
   double minP_;
   double minPCaloMuon_;
   int    minNumberOfMatches_;
   double maxAbsEta_;
   bool   addExtraSoftMuons_;
   
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
   bool writeIsoDeposits_;
   double ptThresholdToFillCandidateP4WithGlobalFit_;
   double sigmaThresholdToFillCandidateP4WithGlobalFit_;
   
   bool debugWithTruthMatching_;

   edm::Handle<reco::TrackCollection>             innerTrackCollectionHandle_;
   edm::Handle<reco::TrackCollection>             outerTrackCollectionHandle_;
   edm::Handle<reco::MuonCollection>              muonCollectionHandle_;
   edm::Handle<reco::MuonTrackLinksCollection>    linkCollectionHandle_;
   edm::Handle<reco::TrackToTrackMap>             tpfmsCollectionHandle_;
   edm::Handle<reco::TrackToTrackMap>             pickyCollectionHandle_;
   edm::Handle<reco::TrackToTrackMap>             dytCollectionHandle_;
   
   MuonCaloCompatibility muonCaloCompatibility_;
   reco::isodeposit::IsoDepositExtractor* muIsoExtractorCalo_;
   reco::isodeposit::IsoDepositExtractor* muIsoExtractorTrack_;
   reco::isodeposit::IsoDepositExtractor* muIsoExtractorJet_;
   std::string trackDepositName_;
   std::string ecalDepositName_;
   std::string hcalDepositName_;
   std::string hoDepositName_;
   std::string jetDepositName_;

   bool          fillGlobalTrackQuality_;
   bool          fillGlobalTrackRefits_ ;
   edm::InputTag globalTrackQualityInputTag_;

   bool fillTrackerKink_;
   std::auto_ptr<MuonKinkFinder> trackerKinkFinder_;

   double caloCut_;
   
   bool arbClean_;
   MuonMesh* meshAlgo_;

};
#endif
