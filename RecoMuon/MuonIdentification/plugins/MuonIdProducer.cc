// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
// 
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonIdProducer.cc,v 1.14 2007/10/06 01:00:42 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include <boost/regex.hpp>
#include "RecoMuon/MuonIdentification/plugins/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/interface/MuonIdTruthInfo.h"
#include "RecoMuon/MuonIdentification/interface/MuonArbitrationMethods.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <algorithm>

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig):
muIsoExtractorCalo_(0),muIsoExtractorTrack_(0)
{
   produces<reco::MuonCollection>();
   
   minPt_                   = iConfig.getParameter<double>("minPt");
   minP_                    = iConfig.getParameter<double>("minP");
   minNumberOfMatches_      = iConfig.getParameter<int>("minNumberOfMatches");
   addExtraSoftMuons_       = iConfig.getParameter<bool>("addExtraSoftMuons");
   maxAbsEta_               = iConfig.getParameter<double>("maxAbsEta");
   maxAbsDx_                = iConfig.getParameter<double>("maxAbsDx");
   maxAbsPullX_             = iConfig.getParameter<double>("maxAbsPullX");
   maxAbsDy_                = iConfig.getParameter<double>("maxAbsDy");
   maxAbsPullY_             = iConfig.getParameter<double>("maxAbsPullY");
   fillCaloCompatibility_   = iConfig.getParameter<bool>("fillCaloCompatibility");
   fillEnergy_              = iConfig.getParameter<bool>("fillEnergy");
   fillMatching_            = iConfig.getParameter<bool>("fillMatching");
   fillIsolation_           = iConfig.getParameter<bool>("fillIsolation");
   
   // Load TrackDetectorAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   
   if (fillCaloCompatibility_){
      // Load MuonCaloCompatibility parameters
      parameters = iConfig.getParameter<edm::ParameterSet>("MuonCaloCompatibility");
      muonCaloCompatibility_.configure( parameters );
   }

   if (fillIsolation_){
      // Load MuIsoExtractor parameters
      edm::ParameterSet caloExtractorPSet = iConfig.getParameter<edm::ParameterSet>("CaloExtractorPSet");
      std::string caloExtractorName = caloExtractorPSet.getParameter<std::string>("ComponentName");
      muIsoExtractorCalo_ = MuIsoExtractorFactory::get()->create( caloExtractorName, caloExtractorPSet);

      edm::ParameterSet trackExtractorPSet = iConfig.getParameter<edm::ParameterSet>("TrackExtractorPSet");
      std::string trackExtractorName = trackExtractorPSet.getParameter<std::string>("ComponentName");
      muIsoExtractorTrack_ = MuIsoExtractorFactory::get()->create( trackExtractorName, trackExtractorPSet);
   }
   
   inputCollectionLabels_ = iConfig.getParameter<std::vector<edm::InputTag> >("inputCollectionLabels");
   inputCollectionTypes_  = iConfig.getParameter<std::vector<std::string> >("inputCollectionTypes");
   if (inputCollectionLabels_.size() != inputCollectionTypes_.size()) 
     throw cms::Exception("ConfigurationError") << "Number of input collection labels is different from number of types. " <<
     "For each collection label there should be exactly one collection type specified.";
   if (inputCollectionLabels_.size()>4 ||inputCollectionLabels_.empty()) 
     throw cms::Exception("ConfigurationError") << "Number of input collections should be from 1 to 4.";
   
   debugWithTruthMatching_    = iConfig.getParameter<bool>("debugWithTruthMatching");
   if (debugWithTruthMatching_) edm::LogWarning("MuonIdentification") 
     << "========================================================================\n" 
     << "Debugging mode with truth matching is turned on!!! Make sure you understand what you are doing!\n"
     << "========================================================================\n";
}


MuonIdProducer::~MuonIdProducer()
{
   if (muIsoExtractorCalo_) delete muIsoExtractorCalo_;
   if (muIsoExtractorTrack_) delete muIsoExtractorTrack_;
   // TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   TimerStack timers;
   timers.push("MuonIdProducer::produce::init");
   
   innerTrackCollectionHandle_.clear();
   outerTrackCollectionHandle_.clear();
   linkCollectionHandle_.clear();
   muonCollectionHandle_.clear();
   
   timers.push("MuonIdProducer::produce::init::getPropagator");
   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
   trackAssociator_.setPropagator(propagator.product());
   
   timers.pop_and_push("MuonIdProducer::produce::init::getInputCollections");
   for ( unsigned int i = 0; i < inputCollectionLabels_.size(); ++i ) {
      if ( inputCollectionTypes_[i] == "inner tracks" ) {
	 iEvent.getByLabel(inputCollectionLabels_[i], innerTrackCollectionHandle_);
	 if (! innerTrackCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input inner tracks: " << innerTrackCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "outer tracks" ) {
	 iEvent.getByLabel(inputCollectionLabels_[i], outerTrackCollectionHandle_);
	 if (! outerTrackCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input outer tracks: " << outerTrackCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "links" ) {
	 iEvent.getByLabel(inputCollectionLabels_[i], linkCollectionHandle_);
	 if (! linkCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input link collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input links: " << linkCollectionHandle_->size();
	 continue;
      }
      if ( inputCollectionTypes_[i] == "muons" ) {
	 iEvent.getByLabel(inputCollectionLabels_[i], muonCollectionHandle_);
	 if (! muonCollectionHandle_.isValid()) 
	   throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputCollectionLabels_[i];
	 LogTrace("MuonIdentification") << "Number of input muons: " << muonCollectionHandle_->size();
	 continue;
      }
      throw cms::Exception("FatalError") << "Unknown input collection type: " << inputCollectionTypes_[i];
   }
}

reco::Muon MuonIdProducer::makeMuon(edm::Event& iEvent, const edm::EventSetup& iSetup, 
				     const reco::TrackRef& track, MuonIdProducer::TrackType type)
{
   LogTrace("MuonIdentification") << "Creating a muon from a track " << track.get()->pt() << 
     " Pt (GeV), eta: " << track.get()->eta();
   reco::Muon aMuon( makeMuon( *(track.get()) ) );
   switch (type) {
    case InnerTrack: 
      aMuon.setTrack( track );
      break;
    case OuterTrack:
      aMuon.setStandAlone( track );
      break;
    case CombinedTrack:
      aMuon.setCombined( track );
      break;
   }
   return aMuon;
}

reco::Muon MuonIdProducer::makeMuon( const reco::MuonTrackLinks& links )
{
   LogTrace("MuonIdentification") << "Creating a muon from a link to tracks object";
   reco::Muon aMuon = makeMuon( *(links.globalTrack()) );
   aMuon.setTrack( links.trackerTrack() );
   aMuon.setStandAlone( links.standAloneTrack() );
   aMuon.setCombined( links.globalTrack() );
   return aMuon;
}

bool MuonIdProducer::isGoodTrack( const reco::Track& track )
{
   // Pt and absolute momentum requirement
   if (track.pt() < minPt_ && track.p() < minP_){ 
      LogTrace("MuonIdentification") << "Skipped low momentum track (Pt,P): " << track.pt() <<
	", " << track.p() << " GeV";
      return false;
   }
	
   // Eta requirement
   if ( fabs(track.eta()) > maxAbsEta_ ){
      LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << track.eta() << " )";
      return false;
   }
   return true;
}
   
unsigned int MuonIdProducer::getChamberId( const DetId& id )
{
   switch ( id.det() ) {
    case DetId::Muon:
      switch ( id.subdetId() ) {
       case MuonSubdetId::DT:
	   { 
	      DTChamberId detId(id.rawId());
	      return detId.rawId();
	   }
	 break;
       case MuonSubdetId::CSC:
	   {
	      CSCDetId detId(id.rawId());
	      return detId.chamberId().rawId();
	   }
	 break;
       default:
	 return 0;
      }
    default:
      return 0;
   }
   return 0;
}


int MuonIdProducer::overlap(const reco::Muon& muon, const reco::Track& track)
{
   int numberOfCommonDetIds = 0;
   if ( ! muon.isMatchesValid() || 
	track.extra().isNull() ||
	track.extra()->recHits().isNull() ) return numberOfCommonDetIds;
   const std::vector<reco::MuonChamberMatch>& matches( muon.getMatches() );
   for ( std::vector<reco::MuonChamberMatch>::const_iterator match = matches.begin();
	 match != matches.end(); ++match ) 
     {
	if ( match->segmentMatches.empty() ) continue;
	bool foundCommonDetId = false;
	
	for ( TrackingRecHitRefVector::const_iterator hit = track.extra()->recHitsBegin();
	      hit != track.extra()->recHitsEnd(); ++hit )
	  {
	     // LogTrace("MuonIdentification") << "hit DetId: " << std::hex << hit->get()->geographicalId().rawId() <<
	     //  "\t hit chamber DetId: " << getChamberId(hit->get()->geographicalId()) <<
	     //  "\t segment DetId: " << match->id.rawId() << std::dec;
	     
	     if ( getChamberId(hit->get()->geographicalId()) == match->id.rawId() ) {
		foundCommonDetId = true;
		break;
	     }
	  }
	if ( foundCommonDetId ) {
	   numberOfCommonDetIds++;
	   break;
	}
     }
   return numberOfCommonDetIds;
}


void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   TimerStack timers;
   timers.push("MuonIdProducer::produce");
   
   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection);
   init(iEvent, iSetup);

   // loop over input collections
   
   // muons first
   if ( muonCollectionHandle_.isValid() )
     for ( reco::MuonCollection::const_iterator muon = muonCollectionHandle_->begin();
	   muon !=  muonCollectionHandle_->end(); ++muon )
       outputMuons->push_back(*muon);
   
   // links second ( assume global muon type )
   if ( linkCollectionHandle_.isValid() )
     for ( reco::MuonTrackLinksCollection::const_iterator links = linkCollectionHandle_->begin();
	   links != linkCollectionHandle_->end(); ++links )
       {
	  // check if this muon is already in the list
	  bool newMuon = true;
	  for ( reco::MuonCollection::const_iterator muon = outputMuons->begin();
		muon !=  outputMuons->end(); ++muon )
	     if ( muon->track() == links->trackerTrack() &&
		  muon->standAloneMuon() == links->standAloneTrack() &&
		  muon->combinedMuon() == links->globalTrack() )
	      newMuon = false;
	  if ( newMuon ) {
	     outputMuons->push_back( makeMuon( *links ) );
	     outputMuons->back().setType(reco::Muon::GlobalMuon | reco::Muon::StandAloneMuon);
	  }
       }
   
   // tracker muon is next
   if ( innerTrackCollectionHandle_.isValid() ) {
      LogTrace("MuonIdentification") << "Creating tracker muons";
      for ( unsigned int i = 0; i < innerTrackCollectionHandle_->size(); ++i )
	{
	   if ( ! isGoodTrack( innerTrackCollectionHandle_->at(i) ) ) continue;
	   
	   // make muon
	   timers.push("MuonIdProducer::produce::fillMuonId");
	   reco::Muon trackerMuon( makeMuon(iEvent, iSetup, reco::TrackRef( innerTrackCollectionHandle_, i ), InnerTrack ) );
	   trackerMuon.setType( reco::Muon::TrackerMuon );
	   fillMuonId(iEvent, iSetup, trackerMuon);
	   if ( ! isGoodTrackerMuon( trackerMuon ) ){
	      LogTrace("MuonIdentification") << "track failed minimal number of muon matches requirement";
	      continue;
	   }
	   timers.pop();
	  
	   if ( debugWithTruthMatching_ ) {
	      // add MC hits to a list of matched segments. 
	      // Since it's debugging mode - code is slow
	      MuonIdTruthInfo::truthMatchMuon(iEvent, iSetup, trackerMuon);
	   }
	  
	   // check if this muon is already in the list
	   bool newMuon = true;
	   for ( reco::MuonCollection::iterator muon = outputMuons->begin();
		 muon !=  outputMuons->end(); ++muon )
	     if ( muon->track().get() == trackerMuon.track().get() ) {
		newMuon = false;
		muon->setMatches( trackerMuon.getMatches() );
		muon->setCalEnergy( trackerMuon.getCalEnergy() );
		muon->setType( muon->getType() | reco::Muon::TrackerMuon );
		LogTrace("MuonIdentification") << "Found a corresponding global muon. Set energy, matches and move on";
		break;
	     }
	   if ( newMuon ) outputMuons->push_back( trackerMuon );
	}
   }
   
   // and at last the stand alone muons
   if ( outerTrackCollectionHandle_.isValid() ) {
      LogTrace("MuonIdentification") << "Looking for new muons among stand alone muon tracks";
      for ( unsigned int i = 0; i < outerTrackCollectionHandle_->size(); ++i )
	{
	   // check if this muon is already in the list of global muons
	   bool newMuon = true;
	   for ( reco::MuonCollection::iterator muon = outputMuons->begin();
		 muon !=  outputMuons->end(); ++muon ) 
	     {
		if ( ! muon->standAloneMuon().isNull() ) {
		   // global muon
		   if ( muon->standAloneMuon().get() ==  &(outerTrackCollectionHandle_->at(i)) ) {
		      newMuon = false;
		      break;
		   }
		} else {
		   // tracker muon - no direct links to the standalone muon
		   // since we have only a few real muons in an event, matching 
		   // the stand alone muon to the tracker muon by DetIds should 
		   // be good enough for association. At the end it's up to a 
		   // user to redefine the association and what it means. Here 
		   // we would like to avoid obvious double counting and we 
		   // tolerate a potential miss association
		   if ( overlap(*muon,outerTrackCollectionHandle_->at(i))>0 ) {
		      LogTrace("MuonIdentification") << "Found associated tracker muon. Set a reference and move on";
		      newMuon = false;
		      muon->setStandAlone( reco::TrackRef( outerTrackCollectionHandle_, i ) );
		      muon->setType( muon->getType() | reco::Muon::StandAloneMuon );
		      break;
		   }
		}
	     }
	   if ( newMuon ) {
	      LogTrace("MuonIdentification") << "No associated stand alone track is found. Making a muon";
	      outputMuons->push_back( makeMuon(iEvent, iSetup, 
					       reco::TrackRef( outerTrackCollectionHandle_, i ), OuterTrack ) );
	      outputMuons->back().setType( reco::Muon::StandAloneMuon );
	   }
	}
   }
   
   LogTrace("MuonIdentification") << "Dress up muons if it's necessary";
   // Fill various information
   for ( reco::MuonCollection::iterator muon = outputMuons->begin(); muon != outputMuons->end(); ++muon )
     {
	// Fill muonID
	timers.push("MuonIdProducer::produce::fillMuonId");
	if ( ( fillMatching_ && ! muon->isMatchesValid() ) || 
	     ( fillEnergy_ && !muon->isEnergyValid() ) ) fillMuonId(iEvent, iSetup, *muon);
	timers.pop();
	
	timers.push("MuonIdProducer::produce::fillCaloCompatibility");
	if ( fillCaloCompatibility_ ) muon->setCaloCompatibility( muonCaloCompatibility_.evaluate(*muon) );
	timers.pop();
	
	timers.push("MuonIdProducer::produce::fillIsolation");
	if ( fillIsolation_ ) fillMuonIsolation(iEvent, iSetup, *muon);
	timers.pop();
     }
	
   LogTrace("MuonIdentification") << "number of muons produced: " << outputMuons->size();
   timers.push("MuonIdProducer::produce::fillArbitration");
   if ( fillMatching_ ) fillArbitrationInfo( outputMuons.get() );
   timers.pop();
   iEvent.put(outputMuons);
}

bool MuonIdProducer::isGoodTrackerMuon( const reco::Muon& muon )
{
   if ( addExtraSoftMuons_ && 
	muon.pt()<5 && fabs(muon.eta())<1.5 && 
	muon.numberOfMatches( reco::Muon::NoArbitration ) >= 1 ) return true;
   return ( muon.numberOfMatches( reco::Muon::NoArbitration ) >= minNumberOfMatches_ );
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::Muon& aMuon)
{
   // perform track - detector association
   const reco::Track* track = 0;
   if ( ! aMuon.track().isNull() )
     track = aMuon.track().get();
   else 
     {
	if ( ! aMuon.standAloneMuon().isNull() )
	  track = aMuon.standAloneMuon().get();
	else
	  throw cms::Exception("FatalError") << "Failed to fill muon id information for a muon with undefined references to tracks"; 
     }

     TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *track, parameters_);
   
   if ( fillEnergy_ ) {
      reco::MuonEnergy muonEnergy;
      muonEnergy.em  = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      muonEnergy.had = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
      muonEnergy.ho  = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
      muonEnergy.emS9  = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits,1); // 3x3 energy
      muonEnergy.hadS9 = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits,1); // 3x3 energy
      muonEnergy.hoS9  = info.nXnEnergy(TrackDetMatchInfo::HORecHits,1);   // 3x3 energy
      aMuon.setCalEnergy( muonEnergy );
   }
   if ( ! fillMatching_ || aMuon.isStandAloneMuon() ) return;
   
   // fill muon match info
   std::vector<reco::MuonChamberMatch> muonChamberMatches;
   unsigned int nubmerOfMatchesAccordingToTrackAssociator = 0;
   for( std::vector<TAMuonChamberMatch>::const_iterator chamber=info.chambers.begin();
	chamber!=info.chambers.end(); chamber++ )
     {
	reco::MuonChamberMatch matchedChamber;
	
	LocalError localError = chamber->tState.localError().positionError();
	matchedChamber.x = chamber->tState.localPosition().x();
	matchedChamber.y = chamber->tState.localPosition().y();
	matchedChamber.xErr = sqrt( localError.xx() );
	matchedChamber.yErr = sqrt( localError.yy() );
	                                                                                                                                                    
	matchedChamber.dXdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().x()/chamber->tState.localDirection().z():9999;
	matchedChamber.dYdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().y()/chamber->tState.localDirection().z():9999;
	// DANGEROUS - compiler cannot guaranty parameters ordering
	AlgebraicSymMatrix55 trajectoryCovMatrix = chamber->tState.localError().matrix();
	matchedChamber.dXdZErr = trajectoryCovMatrix(1,1)>0?sqrt(trajectoryCovMatrix(1,1)):0;
	matchedChamber.dYdZErr = trajectoryCovMatrix(2,2)>0?sqrt(trajectoryCovMatrix(2,2)):0;
	
	matchedChamber.edgeX = chamber->localDistanceX;
	matchedChamber.edgeY = chamber->localDistanceY;
	
	matchedChamber.id = chamber->id;
	if ( ! chamber->segments.empty() ) ++nubmerOfMatchesAccordingToTrackAssociator;
	
	// fill segments
	for( std::vector<TAMuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
	     segment != chamber->segments.end(); segment++ ) 
	  {
	     reco::MuonSegmentMatch matchedSegment;
	     matchedSegment.x = segment->segmentLocalPosition.x();
	     matchedSegment.y = segment->segmentLocalPosition.y();
	     matchedSegment.dXdZ = segment->segmentLocalDirection.x()/segment->segmentLocalDirection.z();
	     matchedSegment.dYdZ = segment->segmentLocalDirection.y()/segment->segmentLocalDirection.z();
	     matchedSegment.xErr = segment->segmentLocalErrorXX>0?sqrt(segment->segmentLocalErrorXX):0;
	     matchedSegment.yErr = segment->segmentLocalErrorYY>0?sqrt(segment->segmentLocalErrorYY):0;
	     matchedSegment.dXdZErr = segment->segmentLocalErrorDxDz>0?sqrt(segment->segmentLocalErrorDxDz):0;
	     matchedSegment.dYdZErr = segment->segmentLocalErrorDyDz>0?sqrt(segment->segmentLocalErrorDyDz):0;
	     matchedSegment.mask = 0;
	     // test segment
	     bool matchedX = false;
	     bool matchedY = false;
	     if (fabs(matchedSegment.x - matchedChamber.x) < maxAbsDx_) matchedX = true;
	     if (fabs(matchedSegment.y - matchedChamber.y) < maxAbsDy_) matchedY = true;
	     if (matchedSegment.xErr>0 && matchedChamber.xErr>0 && 
		 fabs(matchedSegment.x - matchedChamber.x)/sqrt(pow(matchedSegment.xErr,2) + pow(matchedChamber.xErr,2)) < maxAbsPullX_) matchedX = true;
	     if (matchedSegment.yErr>0 && matchedChamber.yErr>0 && 
		 fabs(matchedSegment.y - matchedChamber.y)/sqrt(pow(matchedSegment.yErr,2) + pow(matchedChamber.yErr,2)) < maxAbsPullY_) matchedY = true;
	     if (matchedX && matchedY) matchedChamber.segmentMatches.push_back(matchedSegment);
	  }
	muonChamberMatches.push_back(matchedChamber);
     }
   aMuon.setMatches(muonChamberMatches);
   LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.getMatches().size() << "\n" 
     << "number of chambers with segments according to the associator requirements: " << 
     nubmerOfMatchesAccordingToTrackAssociator;
   LogTrace("MuonIdentification") << "number of segment matches with the producer requirements: " << 
     aMuon.numberOfMatches( reco::Muon::NoArbitration );
}

void MuonIdProducer::fillArbitrationInfo( reco::MuonCollection* pOutputMuons )
{
   //
   // apply segment flags
   //
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > chamberPairs;     // for chamber segment sorting
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > stationPairs;     // for station segment sorting
   std::vector<std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> > arbitrationPairs; // for muon segment arbitration

   // muonIndex1
   for( unsigned int muonIndex1 = 0; muonIndex1 < pOutputMuons->size(); ++muonIndex1 )
   {
      // chamberIter1
      for( std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = pOutputMuons->at(muonIndex1).getMatches().begin();
            chamberIter1 != pOutputMuons->at(muonIndex1).getMatches().end(); ++chamberIter1 )
      {
         if(chamberIter1->segmentMatches.empty()) continue;
         chamberPairs.clear();

         // segmentIter1
         for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
               segmentIter1 != chamberIter1->segmentMatches.end(); ++segmentIter1 )
         {
            chamberPairs.push_back(std::make_pair(&(*chamberIter1), &(*segmentIter1)));
            if(!segmentIter1->isMask()) // has not yet been arbitrated
            {
               arbitrationPairs.clear();
               arbitrationPairs.push_back(std::make_pair(&(*chamberIter1), &(*segmentIter1)));

               // find identical segments with which to arbitrate
               // muonIndex2
               for( unsigned int muonIndex2 = muonIndex1+1; muonIndex2 < pOutputMuons->size(); ++muonIndex2 )
               {
                  // chamberIter2
                  for( std::vector<reco::MuonChamberMatch>::iterator chamberIter2 = pOutputMuons->at(muonIndex2).getMatches().begin();
                        chamberIter2 != pOutputMuons->at(muonIndex2).getMatches().end(); ++chamberIter2 )
                  {
                     // segmentIter2
                     for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter2 = chamberIter2->segmentMatches.begin();
                           segmentIter2 != chamberIter2->segmentMatches.end(); ++segmentIter2 )
                     {
                        if(segmentIter2->isMask()) continue; // has already been arbitrated
                        if(fabs(segmentIter2->x       - segmentIter1->x      ) < 1E-3 &&
                           fabs(segmentIter2->y       - segmentIter1->y      ) < 1E-3 &&
                           fabs(segmentIter2->dXdZ    - segmentIter1->dXdZ   ) < 1E-3 &&
                           fabs(segmentIter2->dYdZ    - segmentIter1->dYdZ   ) < 1E-3 &&
                           fabs(segmentIter2->xErr    - segmentIter1->xErr   ) < 1E-3 &&
                           fabs(segmentIter2->yErr    - segmentIter1->yErr   ) < 1E-3 &&
                           fabs(segmentIter2->dXdZErr - segmentIter1->dXdZErr) < 1E-3 &&
                           fabs(segmentIter2->dYdZErr - segmentIter1->dYdZErr) < 1E-3)
                           arbitrationPairs.push_back(std::make_pair(&(*chamberIter2), &(*segmentIter2)));
                     } // segmentIter2
                  } // chamberIter2
               } // muonIndex2

               // arbitration segment sort
               if(arbitrationPairs.empty()) continue; // this should never happen
               if(arbitrationPairs.size()==1) {
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::Arbitrated);
               } else {
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDRSlope));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDRSlope);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDXSlope));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDXSlope);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDR));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDR);
                  sort(arbitrationPairs.begin(), arbitrationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BelongsToTrackByDX));
                  arbitrationPairs.front().second->setMask(reco::MuonSegmentMatch::BelongsToTrackByDX);
                  for( unsigned int it = 0; it < arbitrationPairs.size(); ++it )
                     arbitrationPairs.at(it).second->setMask(reco::MuonSegmentMatch::Arbitrated);
               }
            }
         } // segmentIter1

         // chamber segment sort
         if(chamberPairs.empty()) continue; // this should never happen
         if(chamberPairs.size()==1) {
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
         } else {
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDRSlope));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDRSlope);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDXSlope));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDXSlope);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDR));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDR);
            sort(chamberPairs.begin(), chamberPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInChamberByDX));
            chamberPairs.front().second->setMask(reco::MuonSegmentMatch::BestInChamberByDX);
         }
      } // chamberIter1

      // station segment sort
      for( int stationIndex = 1; stationIndex < 5; ++stationIndex )
         for( int detectorIndex = 1; detectorIndex < 4; ++detectorIndex )
         {
            stationPairs.clear();

            // chamberIter
            for( std::vector<reco::MuonChamberMatch>::iterator chamberIter = pOutputMuons->at(muonIndex1).getMatches().begin();
                  chamberIter != pOutputMuons->at(muonIndex1).getMatches().end(); ++chamberIter )
            {
               if(!(chamberIter->station()==stationIndex && chamberIter->detector()==detectorIndex)) continue;
               if(chamberIter->segmentMatches.empty()) continue;

               for( std::vector<reco::MuonSegmentMatch>::iterator segmentIter = chamberIter->segmentMatches.begin();
                     segmentIter != chamberIter->segmentMatches.end(); ++segmentIter )
                  stationPairs.push_back(std::make_pair(&(*chamberIter), &(*segmentIter)));
            } // chamberIter

            if(stationPairs.empty()) continue; // this may very well happen
            if(stationPairs.size()==1) {
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
            } else {
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDRSlope));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDRSlope);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDXSlope));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDXSlope);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDR));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDR);
               sort(stationPairs.begin(), stationPairs.end(), SortMuonSegmentMatches(reco::MuonSegmentMatch::BestInStationByDX));
               stationPairs.front().second->setMask(reco::MuonSegmentMatch::BestInStationByDX);
            }
         }

   } // muonIndex1
}

void MuonIdProducer::fillMuonIsolation(edm::Event& iEvent, const edm::EventSetup& iSetup, reco::Muon& aMuon)
{
   reco::MuonIsolation isoR03, isoR05;
   const reco::Track* track = 0;
   if ( ! aMuon.track().isNull() )
     track = aMuon.track().get();
   else 
     {
	if ( ! aMuon.standAloneMuon().isNull() )
	  track = aMuon.standAloneMuon().get();
	else
	  throw cms::Exception("FatalError") << "Failed to compute muon isolation information for a muon with undefined references to tracks"; 
     }

   // get deposits
   reco::MuIsoDeposit depTrk = muIsoExtractorTrack_->deposit(iEvent, iSetup, *track );
   std::vector<reco::MuIsoDeposit> caloDeps = muIsoExtractorCalo_->deposits(iEvent, iSetup, *track);

   if(caloDeps.size()!=3) {
      LogTrace("MuonIdentification") << "Failed to fill vector of calorimeter isolation deposits!";
      return;
   }

   reco::MuIsoDeposit depEcal = caloDeps.at(0);
   reco::MuIsoDeposit depHcal = caloDeps.at(1);
   reco::MuIsoDeposit depHo   = caloDeps.at(2);

   isoR03.sumPt     = depTrk.depositWithin(0.3);
   isoR03.emEt      = depEcal.depositWithin(0.3);
   isoR03.hadEt     = depHcal.depositWithin(0.3);
   isoR03.hoEt      = depHo.depositWithin(0.3);
   isoR03.nTracks   = depTrk.depositAndCountWithin(0.3).second;
   isoR03.nJets     = 0;

   isoR05.sumPt     = depTrk.depositWithin(0.5);
   isoR05.emEt      = depEcal.depositWithin(0.5);
   isoR05.hadEt     = depHcal.depositWithin(0.5);
   isoR05.hoEt      = depHo.depositWithin(0.5);
   isoR05.nTracks   = depTrk.depositAndCountWithin(0.5).second;
   isoR05.nJets     = 0;

   aMuon.setIsolation(isoR03, isoR05);
}

reco::Muon MuonIdProducer::makeMuon( const reco::Track& track )
{
   //FIXME: E = sqrt(p^2 + m^2), where m == 0.105658369(9)GeV 
   double energy = sqrt(track.p() * track.p() + 0.011163691);
   math::XYZTLorentzVector p4(track.px(),
			      track.py(),
			      track.pz(),
			      energy);
   return reco::Muon( track.charge(), p4, track.vertex() );
}
