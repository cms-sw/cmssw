// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonIdProducer
// 
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonIdProducer.cc,v 1.5 2007/06/08 17:25:44 dmytro Exp $
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

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig):
muIsoExtractorCalo_(0),muIsoExtractorTrack_(0)
{
   branchAlias_ = iConfig.getParameter<std::string>("branchAlias");
   produces<reco::MuonCollection>().setBranchAlias(branchAlias_);
   
   minPt_                   = iConfig.getParameter<double>("minPt");
   minP_                    = iConfig.getParameter<double>("minP");
   minNumberOfMatches_      = iConfig.getParameter<int>("minNumberOfMatches");
   stiffMinPt_              = iConfig.getParameter<double>("stiffMinPt");
   stiffMinP_               = iConfig.getParameter<double>("stiffMinP");
   stiffMinNumberOfMatches_ = iConfig.getParameter<int>("stiffMinNumberOfMatches");
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
   
   inputTrackCollectionLabel_ = iConfig.getParameter<edm::InputTag>("inputTrackCollection");
   inputMuonCollectionLabel_  = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   inputLinkCollectionLabel_  = iConfig.getParameter<edm::InputTag>("inputLinkCollection");
   int mode                   = iConfig.getParameter<int>("inputType");
   if (mode>3||mode<0) throw cms::Exception("ConfigurationError") << "Unsupported input collection type: " << mode_;
   mode_ = InputMode(mode);
   
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
   TimingReport::current()->dump(std::cout);
}

void MuonIdProducer::init(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   TimerStack timers;
   timers.push("MuonIdProducer::produce::init");
   timers.push("MuonIdProducer::produce::init::getPropagator");
   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
   trackAssociator_.setPropagator(propagator.product());
   
   timers.pop_and_push("MuonIdProducer::produce::init::getInputCollections");
   try { iEvent.getByLabel(inputTrackCollectionLabel_, trackCollectionHandle_); } catch(...){} ;
   try { iEvent.getByLabel(inputMuonCollectionLabel_, muonCollectionHandle_); } catch(...){} ;
   try { iEvent.getByLabel(inputLinkCollectionLabel_, linkCollectionHandle_); } catch(...){} ;
   
   timers.pop_and_push("MuonIdProducer::produce::init::misc");
   switch ( mode_ ){
    case TrackCollection:
      if (! trackCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Failed to get input track collection with label: " << inputTrackCollectionLabel_;
      LogTrace("MuonIdentification") << "Number of input tracks: " << trackCollectionHandle_->size();
      trackCollectionIter_ = trackCollectionHandle_->begin();
      index_ = 0;
      break;
    case MuonCollection:
      if (! muonCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Failed to get input muon collection with label: " << inputMuonCollectionLabel_; 
      LogTrace("MuonIdentification") << "Number of input muons: " << muonCollectionHandle_->size();
      muonCollectionIter_ = muonCollectionHandle_->begin();
      break;
    case LinkCollection:
      if (! linkCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Failed to get input muon-track link collection with label: " << inputLinkCollectionLabel_;
      LogTrace("MuonIdentification") << "Number of input links: " << linkCollectionHandle_->size();
      linkCollectionIter_ = linkCollectionHandle_->begin();
      break;
   }
}

reco::Muon* MuonIdProducer::nextMuon(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   reco::Muon* aMuon = 0;
   switch (mode_) {
    case TrackCollection:
      if( trackCollectionIter_ !=  trackCollectionHandle_->end()) {
	 LogTrace("MuonIdentification") << "Creating a muon from a track";
	 aMuon = makeMuon(*trackCollectionIter_);
	 aMuon->setTrack( reco::TrackRef( trackCollectionHandle_, index_ ) );

	 // loop over muons or links to fill the missing references to tracks
	 if ( muonCollectionHandle_.isValid() ) {
	    for( reco::MuonCollection::const_iterator muon = muonCollectionHandle_->begin();
		 muon != muonCollectionHandle_->end(); ++muon )
	      if ( muon->track().id() == aMuon->track().id() ) {
		 aMuon->setStandAlone(muon->standAloneMuon());
		 aMuon->setCombined(muon->combinedMuon());
		 break;
	      }
	 } else {
	    if ( linkCollectionHandle_.isValid() )
	      for( reco::MuonTrackLinksCollection::const_iterator link = linkCollectionHandle_->begin();
		   link != linkCollectionHandle_->end(); ++link )
		if ( link->trackerTrack().id() == aMuon->track().id() ) {
		   aMuon->setStandAlone(link->standAloneTrack());
		   aMuon->setCombined(link->globalTrack());
		   break;
		}
	 }
	 index_++;
	 trackCollectionIter_++;
      } 
      break;
    case MuonCollection:
      if( muonCollectionIter_ !=  muonCollectionHandle_->end())	{
	 LogTrace("MuonIdentification") << "Creating a muon from a muon";
	 aMuon = new reco::Muon(*muonCollectionIter_);
	 muonCollectionIter_++;
      }
      break;
    case LinkCollection:
      if( linkCollectionIter_ !=  linkCollectionHandle_->end())	{
	 LogTrace("MuonIdentification") << "Creating a muon from a link to tracks object";
	 aMuon = makeMuon( *(linkCollectionIter_->globalTrack()) );
	 aMuon->setTrack(linkCollectionIter_->trackerTrack());
	 aMuon->setStandAlone(linkCollectionIter_->standAloneTrack());
	 aMuon->setCombined(linkCollectionIter_->globalTrack());
	 linkCollectionIter_++;
      }
      break;
   }
   return aMuon;
}

void MuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection);

   TimerStack timers;
   timers.push("MuonIdProducer::produce");
   init(iEvent, iSetup);

   // loop over input collection
   while(reco::Muon* aMuon = nextMuon(iEvent, iSetup) )
     {
	if ( ! aMuon || ! aMuon->track().get() ) {
	   edm::LogError("MuonIdentification") << "failed to get a valid Muon object. Skip event";
	   break;
	}
	LogTrace("MuonIdentification") << "---------------------------------------------\n" <<
	  "track Pt: " << aMuon->track().get()->pt() << " GeV\n" <<
	  "Distance from IP: " <<  aMuon->track().get()->vertex().rho() << " cm";
	
	std::auto_ptr<reco::Muon> muon(aMuon); // transfer ownership
	
	// Pt requirement
	if (muon->track().get()->pt() < minPt_){ 
	   LogTrace("MuonIdentification") << "Skipped low Pt track (Pt: " << muon->track().get()->pt() << " GeV)";
	   continue;
	}
	
	// Absolute momentum requirement
	if (muon->track().get()->p() < minP_){
	   LogTrace("MuonIdentification") << "Skipped low P track (P: " << muon->track().get()->p() << " GeV)";
	   continue;
	}
	
	// Eta requirement
	if ( fabs(muon->track().get()->eta()) > maxAbsEta_ ){
	   LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << muon->track().get()->eta() << " )";
	   continue;
	}
	
	// Fill muonID
	timers.push("MuonIdProducer::produce::fillMuonId");
	if ( fillMatching_ || fillEnergy_) fillMuonId(iEvent, iSetup, *muon);
	timers.pop();
	   
	bool stiffMuon = (muon->track().get()->pt() > stiffMinPt_ ) && ( muon->track().get()->p() > stiffMinP_ );
	
	if (stiffMuon) LogTrace("MuonIdentification") << "Muon is stiff";
	
	// check number of matches
	if ( fillMatching_ && ( minNumberOfMatches_>0 || 
			       ( stiffMuon && stiffMinNumberOfMatches_>0 ) ) ) {
	   int numberOfMatches = 0;
	   const std::vector<reco::MuonChamberMatch>& chambers = muon->getMatches();
	   for( std::vector<reco::MuonChamberMatch>::const_iterator chamber=chambers.begin(); 
		chamber!=chambers.end(); ++chamber )
	     {
		bool matchedX = false;
		bool matchedY = false;
		for( std::vector<reco::MuonSegmentMatch>::const_iterator segment=chamber->segmentMatches.begin(); 
		     segment!=chamber->segmentMatches.end(); ++segment )
		  {
		     if (fabs(segment->x - chamber->x) < maxAbsDx_) matchedX = true;
		     if (fabs(segment->y - chamber->y) < maxAbsDy_) matchedY = true;
		     if (segment->xErr>0 && chamber->xErr>0 && 
			 fabs(segment->x - chamber->x)/sqrt(pow(segment->xErr,2) + pow(chamber->xErr,2)) < maxAbsPullX_) matchedX = true;
		     if (segment->yErr>0 && chamber->yErr>0 && 
			 fabs(segment->y - chamber->y)/sqrt(pow(segment->yErr,2) + pow(chamber->yErr,2)) < maxAbsPullY_) matchedY = true;
		     if (matchedX && matchedY) break;
		  }
		if ( matchedX && matchedY ) numberOfMatches++;
	     }
	   if ( ( ! stiffMuon && numberOfMatches < minNumberOfMatches_ ) ||
		( ! stiffMuon && numberOfMatches < stiffMinNumberOfMatches_ ) ) continue;
	}
	
	if ( debugWithTruthMatching_ ) {
	   // add MC hits to a list of matched segments. 
	   // Since it's debugging mode - code is slow
	   MuonIdTruthInfo::truthMatchMuon(iEvent, iSetup, *muon);
	}
	
	timers.push("MuonIdProducer::produce::fillCaloCompatibility");
	if ( fillCaloCompatibility_ ) muon->setCaloCompatibility( muonCaloCompatibility_.evaluate(*muon) );
	timers.pop();
	
	timers.push("MuonIdProducer::produce::fillIsolation");
	if ( fillIsolation_ ) fillMuonIsolation(iEvent, iSetup, *muon);
	timers.pop();
	
	outputMuons->push_back(*muon);
     }
   LogTrace("MuonIdentification") << "number of muons produced: " << outputMuons->size();
   timers.push("MuonIdProducer::produce::fillArbitration");
   if ( fillMatching_ ) fillArbitrationInfo( outputMuons.get() );
   timers.pop();
   iEvent.put(outputMuons);
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::Muon& aMuon)
{
   // perform track - detector association
   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *(aMuon.track().get()), parameters_);
   
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
   if ( fillMatching_ ) {
      std::vector<reco::MuonChamberMatch> muonChamberMatches;
      for( std::vector<MuonChamberMatch>::const_iterator chamber=info.chambers.begin();
	   chamber!=info.chambers.end(); chamber++ )
	{
	   reco::MuonChamberMatch aMatch;
	
	   LocalError localError = chamber->tState.localError().positionError();
	   aMatch.x = chamber->tState.localPosition().x();
	   aMatch.y = chamber->tState.localPosition().y();
	   aMatch.xErr = sqrt( localError.xx() );
	   aMatch.yErr = sqrt( localError.yy() );
	                                                                                                                                                    
	   aMatch.dXdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().x()/chamber->tState.localDirection().z():9999;
	   aMatch.dYdZ = chamber->tState.localDirection().z()!=0?chamber->tState.localDirection().y()/chamber->tState.localDirection().z():9999;
	   // DANGEROUS - compiler cannot guaranty parameters ordering
	   AlgebraicSymMatrix55 trajectoryCovMatrix = chamber->tState.localError().matrix();
	   aMatch.dXdZErr = trajectoryCovMatrix(1,1);
	   aMatch.dYdZErr = trajectoryCovMatrix(2,2);
	
	   aMatch.edgeX = chamber->localDistanceX;
	   aMatch.edgeY = chamber->localDistanceY;
	
	   aMatch.id = chamber->id;
	
	   // fill segments
	   for( std::vector<MuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
		segment != chamber->segments.end(); segment++ ) 
	     {
		reco::MuonSegmentMatch aSegment;
		aSegment.x = segment->segmentLocalPosition.x();
		aSegment.y = segment->segmentLocalPosition.y();
		aSegment.dXdZ = segment->segmentLocalDirection.x()/segment->segmentLocalDirection.z();
		aSegment.dYdZ = segment->segmentLocalDirection.y()/segment->segmentLocalDirection.z();
		aSegment.xErr = segment->segmentLocalErrorXX>0?sqrt(segment->segmentLocalErrorXX):0;
		aSegment.yErr = segment->segmentLocalErrorYY>0?sqrt(segment->segmentLocalErrorYY):0;
		aSegment.dXdZErr = segment->segmentLocalErrorDxDz>0?sqrt(segment->segmentLocalErrorDxDz):0;
		aSegment.dYdZErr = segment->segmentLocalErrorDyDz>0?sqrt(segment->segmentLocalErrorDyDz):0;
		aSegment.mask = 0;
		aMatch.segmentMatches.push_back(aSegment);
	     }
	   muonChamberMatches.push_back(aMatch);
	}
      aMuon.setMatches(muonChamberMatches);
   }
   LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.getMatches().size() << "\n" 
     << "number of muon matches: " << aMuon.numberOfMatches();
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
   // get deposits
   reco::MuIsoDeposit depTrk = muIsoExtractorTrack_->deposit(iEvent, iSetup, *(aMuon.track()));
   std::vector<reco::MuIsoDeposit> caloDeps = muIsoExtractorCalo_->deposits(iEvent, iSetup, *(aMuon.track()));

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

reco::Muon* MuonIdProducer::makeMuon( const reco::Track& track )
{
   //FIXME: E = sqrt(p^2 + m^2), where m == 0.105658369(9)GeV 
   double energy = sqrt(track.p() * track.p() + 0.011163691);
   math::XYZTLorentzVector p4(track.px(),
			      track.py(),
			      track.pz(),
			      energy);
   return new reco::Muon( track.charge(), p4, track.vertex() );
}
