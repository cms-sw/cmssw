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
// $Id: MuonIdProducer.cc,v 1.12 2007/04/13 06:13:45 dmytro Exp $
//
//


// system include files
#include <memory>

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

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include <boost/regex.hpp>
#include "RecoMuon/MuonIdentification/interface/MuonIdProducer.h"
#include "RecoMuon/MuonIdentification/interface/MuonIdTruthInfo.h"
#include "RecoMuon/MuonIdentification/interface/MuonArbitrationMethods.h"

#include <algorithm>

MuonIdProducer::MuonIdProducer(const edm::ParameterSet& iConfig)
{
   branchAlias_ = iConfig.getParameter<std::string>("branchAlias");
   produces<reco::MuonCollection>().setBranchAlias(branchAlias_);
   
   minPt_ = iConfig.getParameter<double>("minPt");
   minP_ = iConfig.getParameter<double>("minP");
   maxAbsEta_ = iConfig.getParameter<double>("maxAbsEta");
   minNumberOfMatches_ = iConfig.getParameter<int>("minNumberOfMatches");
   maxAbsDx_ = iConfig.getParameter<double>("maxAbsDx");
   maxAbsPullX_ = iConfig.getParameter<double>("maxAbsPullX");
   maxAbsDy_ = iConfig.getParameter<double>("maxAbsDy");
   maxAbsPullY_ = iConfig.getParameter<double>("maxAbsPullY");
   // Load TrackDetectorAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );

   inputTrackCollectionLabel_ = iConfig.getParameter<edm::InputTag>("inputTrackCollection");
   inputMuonCollectionLabel_  = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   if ( iConfig.getParameter<bool>("useMuonCollectionAsInput") ) mode_ = MuonCollection;
   else mode_ = TrackCollection;

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
   if ( mode_ == TrackCollection ) {
      iEvent.getByLabel(inputTrackCollectionLabel_, trackCollectionHandle_);
      if (! trackCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Cannot find input track collection with label: " << inputTrackCollectionLabel_;
      trackCollectionIter_ = trackCollectionHandle_->begin();
      index_ = 0;
   }else{
      iEvent.getByLabel(inputMuonCollectionLabel_, muonCollectionHandle_);
      if (! muonCollectionHandle_.isValid()) 
	throw cms::Exception("FatalError") << "Cannot find input muon collection with label: " << inputMuonCollectionLabel_; 
      muonCollectionIter_ = muonCollectionHandle_->begin();
   }
}

reco::Muon* MuonIdProducer::getNewMuon(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   switch (mode_) {
    case TrackCollection:
      if( trackCollectionIter_ !=  trackCollectionHandle_->end())
	{
	   reco::Muon* aMuon = new reco::Muon;
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
	   reco::Muon* aMuon = new reco::Muon; // here should be constructor based on reco::Muon
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
   
   std::auto_ptr<reco::MuonCollection> outputMuons(new reco::MuonCollection);

   TimerStack timers;
   timers.push("MuonIdProducer::produce::init");
   init(iEvent, iSetup);
   timers.clear_stack();

   // loop over input collection
   while(reco::Muon* aMuon = getNewMuon(iEvent, iSetup))
     {
	if ( ! aMuon || ! aMuon->track().get() ) {
	   edm::LogError("MuonIdentification") << "failed to make a valid Muon object. Skip event";
	   break;
	}
	LogTrace("MuonIdentification") << "---------------------------------------------";
	LogTrace("MuonIdentification") << "track Pt: " << aMuon->track().get()->pt() << " GeV";
	LogTrace("MuonIdentification") << "Distance from IP: " <<  aMuon->track().get()->vertex().rho() << " cm";
	
	bool goodMuonCandidate = true;
	
	// Pt requirement
	if (goodMuonCandidate && aMuon->track().get()->pt() < minPt_){ 
	   LogTrace("MuonIdentification") << "Skipped low Pt track (Pt: " << aMuon->track().get()->pt() << " GeV)";
	   goodMuonCandidate = false;
	}
	
	// Absolute momentum requirement
	if (goodMuonCandidate && aMuon->track().get()->p() < minP_){
	   LogTrace("MuonIdentification") << "Skipped low P track (P: " << aMuon->track().get()->p() << " GeV)";
	   goodMuonCandidate = false;
	}
	
	// Eta requirement
	if ( goodMuonCandidate && fabs(aMuon->track().get()->eta()) > maxAbsEta_ ){
	   LogTrace("MuonIdentification") << "Skipped track with large pseudo rapidity (Eta: " << aMuon->track().get()->eta() << " )";
	   goodMuonCandidate = false;
	}
	
	// Fill muonID
	if ( goodMuonCandidate ) fillMuonId(iEvent, iSetup, *aMuon);
	   
	// check number of matches
	if ( goodMuonCandidate && minNumberOfMatches_>0) {
	   int numberOfMatches = 0;
	   const std::vector<reco::MuonChamberMatch>& chambers = aMuon->getMatches();
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
	   if (numberOfMatches < minNumberOfMatches_) goodMuonCandidate = false;
	}
	
	if ( goodMuonCandidate && debugWithTruthMatching_ ) {
	   // add MC hits to a list of matched segments. 
	   // Since it's debugging mode - code is slow
	   MuonIdTruthInfo::truthMatchMuon(iEvent, iSetup, *aMuon);
	}
	
	if (goodMuonCandidate ) outputMuons->push_back(*aMuon);
	
	delete aMuon;
     }

   //
   // apply segment flags
   //
   reco::MuonCollection* pOutputMuons = outputMuons.get();
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
   iEvent.put(outputMuons);
}

void MuonIdProducer::fillMuonId(edm::Event& iEvent, const edm::EventSetup& iSetup,
				reco::Muon& aMuon)
{
   TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, 
						       trackAssociator_.getFreeTrajectoryState(iSetup, *(aMuon.track().get()) ),
						       parameters_);
   reco::Muon::MuonEnergy muonEnergy;
   muonEnergy.em  = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
   muonEnergy.had = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
   muonEnergy.ho  = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
      
   aMuon.setCalEnergy( muonEnergy );
      
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
   LogTrace("MuonIdentification") << "number of muon chambers: " << aMuon.getMatches().size() << "\n" 
     << "number of muon matches: " << aMuon.numberOfMatches();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdProducer);
