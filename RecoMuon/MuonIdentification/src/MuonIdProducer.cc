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
// $Id: MuonIdProducer.cc,v 1.7 2007/01/30 18:25:11 dmytro Exp $
//
//


// system include files
#include <memory>

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

#include <boost/regex.hpp>
#include "RecoMuon/MuonIdentification/interface/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/interface/MuonIdTruthInfo.h"

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig)
{
   outputCollectionName_ = iConfig.getParameter<std::string>("outputCollection");
   produces<reco::MuonWithMatchInfoCollection>(outputCollectionName_);

   useEcal_ = true;
   useMuon_ = true;
   useHcalRecHits_ = iConfig.getParameter<bool>("useHcalRecHits");
   
   useOldMuonMatching_ = iConfig.getParameter<bool>("useOldMuonMatching");
   
   minPt_ = iConfig.getParameter<double>("minPt");
   minP_ = iConfig.getParameter<double>("minP");
   maxAbsEta_ = iConfig.getParameter<double>("maxAbsEta");
   minNumberOfMatches_ = iConfig.getParameter<int>("minNumberOfMatches");
   maxAbsDx_ = iConfig.getParameter<double>("maxAbsDx");
   maxAbsPullX_ = iConfig.getParameter<double>("maxAbsPullX");
   maxAbsDy_ = iConfig.getParameter<double>("maxAbsDy");
   maxAbsPullY_ = iConfig.getParameter<double>("maxAbsPullY");
   ecalPreselectionCone_ = iConfig.getParameter<double>("ecalPreselectionCone");
   // ecalSelectionCone_ = iConfig.getParameter<double>("ecalSelectionCone");
   hcalPreselectionCone_ = iConfig.getParameter<double>("hcalPreselectionCone");
   // hcalSelectionCone_ = iConfig.getParameter<double>("hcalSelectionCone");
   muonPreselectionCone_ = iConfig.getParameter<double>("muonPreselectionCone");
   // muonSelectionCone_ = iConfig.getParameter<double>("muonSelectionCone");
   
   // Fill data labels
   trackAssociator_.theEBRecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
   trackAssociator_.theEERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EERecHitCollectionLabel");
   trackAssociator_.theCaloTowerCollectionLabel = iConfig.getParameter<edm::InputTag>("CaloTowerCollectionLabel");
   trackAssociator_.theHBHERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   trackAssociator_.theHORecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   trackAssociator_.theDTRecSegment4DCollectionLabel = iConfig.getParameter<edm::InputTag>("DTRecSegment4DCollectionLabel");
   trackAssociator_.theCSCSegmentCollectionLabel = iConfig.getParameter<edm::InputTag>("CSCSegmentCollectionLabel");

   inputCollectionLabel_ = iConfig.getParameter<edm::InputTag>("inputCollectionLabel");
   inputTypeIsTrack_ =  iConfig.getParameter<bool>("inputTypeIsTrack");

   debugWithTruthMatching_ = iConfig.getParameter<bool>("debugWithTruthMatching");
   if (debugWithTruthMatching_) edm::LogWarning("MuonIdentification") 
     << "========================================================================\n" 
     << "Debugging mode with truth matching is turned on!!! Make sure you understand what you are doing!\n"
     << "========================================================================\n";
   trackAssociator_.useDefaultPropagator();
}


MuonIdProducer::~MuonIdProducer()
{
  TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if ( inputTypeIsTrack_ ) {
      iEvent.getByLabel(inputCollectionLabel_, trackCollectionHandle_);
      if (! trackCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Cannot find input track collection with label: " << inputCollectionLabel_; 
      mode_ = TrackCollection;
      trackCollectionIter_ = trackCollectionHandle_->begin();
      index_ = 0;
   }else{
      iEvent.getByLabel(inputCollectionLabel_, muonCollectionHandle_);
      if (! muonCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Cannot find input muon collection with label: " << inputCollectionLabel_; 
      mode_ = MuonCollection;
      muonCollectionIter_ = muonCollectionHandle_->begin();
   }
}

reco::MuonWithMatchInfo* MuonIdProducer::getNewMuon(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   switch (mode_) {
    case TrackCollection:
      if( trackCollectionIter_ !=  trackCollectionHandle_->end())
	{
	   reco::MuonWithMatchInfo* aMuon = new reco::MuonWithMatchInfo;
	   aMuon->setTrack(reco::TrackRef(trackCollectionHandle_,index_));
	   index_++;
	   trackCollectionIter_++;
	   return aMuon;
	}
      else return 0;
      break;
    case MuonCollection:
      if( muonCollectionIter_ !=  muonCollectionHandle_->end())
	{
	   reco::MuonWithMatchInfo* aMuon = new reco::MuonWithMatchInfo; // here should be constructor based on reco::Muon
	   aMuon->setTrack(muonCollectionIter_->track());
	   aMuon->setStandAlone(muonCollectionIter_->standAloneMuon());
	   aMuon->setCombined(muonCollectionIter_->combinedMuon());
	   muonCollectionIter_++;
	   return aMuon;
	}
      else return 0;
      break;
   }
   return 0;
}

void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::auto_ptr<reco::MuonWithMatchInfoCollection> outputMuons(new reco::MuonWithMatchInfoCollection);

   TimerStack timers;
   timers.push("MuonIdProducer::produce::init");
   init(iEvent, iSetup);
   timers.clean_stack();

   // loop over input collection
   while(reco::MuonWithMatchInfo* aMuon = getNewMuon(iEvent, iSetup))
     {
	if ( ! aMuon || ! aMuon->track().get() ) {
	   edm::LogError("MuonIdentification") << "failed to make a valid MuonWithMatchInfo object. Skip event";
	   break;
	}
	LogTrace("MuonIdentification") << "---------------------------------------------";
	LogTrace("MuonIdentification") << "track Pt: " << aMuon->track().get()->pt() << " GeV";
	LogTrace("MuonIdentification") << "Distance from IP: " <<  aMuon->track().get()->vertex().rho() << " cm";
	
	bool goodMuonCandidate = true;
	
	// Pt requirement
	if (aMuon->track().get()->pt() < minPt_){ 
	   LogTrace("MuonIdentification") << "Skipped low Pt track (Pt: " << aMuon->track().get()->pt() << " GeV)";
	   goodMuonCandidate = false;
	}
	
	// Absolute momentum requirement
	if (aMuon->track().get()->p() < minP_){
	   LogTrace("MuonIdentification") << "Skipped low P track (P: " << aMuon->track().get()->p() << " GeV)";
	   goodMuonCandidate = false;
	}
	
	// Eta requirement
	if ( fabs(aMuon->track().get()->eta()) > maxAbsEta_ ){
	   LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << aMuon->track().get()->eta() << " )";
	   goodMuonCandidate = false;
	}
	
	if ( goodMuonCandidate ){
	   fillMuonId(iEvent, iSetup, *aMuon);
	   
	   // loop over matches
	   
	}
	
	if ( goodMuonCandidate && debugWithTruthMatching_ ) {
	   // add MC hits to a list of matched segments. The only
	   // way to differentiate hits is the error on the local
	   // hit position. It's -9999 for a MC hit.
	   // Since it's debugging mode - code is slow
	   MuonIdTruthInfo::truthMatchMuon(iEvent, iSetup, *aMuon);
	}
	
	if (goodMuonCandidate ) outputMuons->push_back(*aMuon);
	
	delete aMuon;
     }
   iEvent.put(outputMuons,outputCollectionName_);
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::MuonWithMatchInfo& aMuon)
{
   TrackDetectorAssociator::AssociatorParameters parameters;
   parameters.useEcal = useEcal_ ;
   parameters.useHcal = useHcalRecHits_ ;
   parameters.useHO   = useHcalRecHits_ ;
   parameters.useCalo = ! useHcalRecHits_ ;
   parameters.useMuon = useMuon_ ;
   parameters.useOldMuonMatching = useOldMuonMatching_ ;
   
   parameters.dREcalPreselection = ecalPreselectionCone_;
   // parameters.dREcal = ecalSelectionCone_;  TEMPORARY
   parameters.dREcal = ecalPreselectionCone_;
   parameters.dRHcalPreselection = hcalPreselectionCone_;
   parameters.dRHcal = hcalSelectionCone_;
   // parameters.dRHcal = hcalSelectionCone_;  TEMPORARY
   parameters.dRMuonPreselection = muonPreselectionCone_;
   // parameters.dRMuon = muonSelectionCone_;
   parameters.dRMuon = muonPreselectionCone_;

   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, 
						       trackAssociator_.getFreeTrajectoryState(iSetup, *(aMuon.track().get()) ),
						       parameters);
   reco::MuonWithMatchInfo::MuonEnergy muonEnergy;
   muonEnergy.em = info.ecalEnergy();
   if (useHcalRecHits_){
      muonEnergy.had = info.hcalEnergy();
      muonEnergy.ho = info.hoEnergy();
   }else{
      muonEnergy.had = info.hcalTowerEnergy();
      muonEnergy.ho = info.hoTowerEnergy();
   }
      
   aMuon.setCalEnergy( muonEnergy );
      
   std::vector<reco::MuonWithMatchInfo::MuonChamberMatch> muonChamberMatches;
   for( std::vector<MuonChamberMatch>::const_iterator chamber=info.chambers.begin();
	chamber!=info.chambers.end(); chamber++ )
     {
	reco::MuonWithMatchInfo::MuonChamberMatch aMatch;
	
	LocalError localError = chamber->tState.localError().positionError();
	aMatch.x = chamber->tState.localPosition().x();
	aMatch.y = chamber->tState.localPosition().y();
	aMatch.xErr = sqrt( localError.xx() );
	aMatch.yErr = sqrt( localError.yy() );
	                                                                                                                                                    
	aMatch.dXdZ = chamber->tState.localDirection().x();
	aMatch.dYdZ = chamber->tState.localDirection().y();
	// DANGEROUS - compiler cannot guaranty parameters ordering
	AlgebraicSymMatrix trajectoryCovMatrix = chamber->tState.localError().matrix();
	aMatch.dXdZErr = trajectoryCovMatrix[1][1];
	aMatch.dYdZErr = trajectoryCovMatrix[2][2];
	
	aMatch.edgeX = chamber->localDistanceX;
	aMatch.edgeY = chamber->localDistanceY;
	
	aMatch.id = chamber->id;
	
	// fill segments
	for( std::vector<MuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
	     segment != chamber->segments.end(); segment++ ) 
	  {
	     reco::MuonWithMatchInfo::MuonSegmentMatch aSegment;
	     aSegment.x = segment->segmentLocalPosition.x();
	     aSegment.y = segment->segmentLocalPosition.y();
	     aSegment.dXdZ = segment->segmentLocalDirection.x()/segment->segmentLocalDirection.z();
	     aSegment.dYdZ = segment->segmentLocalDirection.y()/segment->segmentLocalDirection.z();
	     aSegment.xErr = segment->segmentLocalErrorXX>0?sqrt(segment->segmentLocalErrorXX):0;
	     aSegment.yErr = segment->segmentLocalErrorYY>0?sqrt(segment->segmentLocalErrorYY):0;
	     aSegment.dXdZErr = segment->segmentLocalErrorDxDz>0?sqrt(segment->segmentLocalErrorDxDz):0;
	     aSegment.dYdZErr = segment->segmentLocalErrorDyDz>0?sqrt(segment->segmentLocalErrorDyDz):0;
	     aMatch.segmentMatches.push_back(aSegment);
	  }
	muonChamberMatches.push_back(aMatch);
     }
   aMuon.setMatches(muonChamberMatches);
   LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.matches().size() << "\n" 
     << "number of muon matches: " << aMuon.numberOfMatches();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdProducer);
