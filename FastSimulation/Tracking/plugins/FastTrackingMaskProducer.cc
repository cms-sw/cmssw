// -*- C++ -*-                                                                                                                   
//                                                                                                                               
// Package:    FastSimulation/FastTrackingMaskProducer                                                                           
// Class:      FastTrackingMaskProducer                                                                                                
//                                                                                                                               
/**\class  FastTrackingMaskProducer FastTrackingMaskProducer.cc FastSimulation/Tracking/plugins/FastTrackingMaskProducer.cc                                                                                                                                
 Description: The class creates two vectors with booleans - hitMasks and hitCombinationMasks. Both of the vectors are filled
 with 'false' values unless the id of a specific rechit has to be masked. The number of entry inside a vector represents the
 id of a hit.
  
 TODO: Consider implementing Chi2 method ascit is done in the FullSim                                                                                                                                                                                             
*/
//                                                                                                                                
// Original Author:  Vilius Kripas                                                                                                
//         Created:  Mon, 22 Jun 2015 15:08:57 GMT                                                                                
//                                                                                                                                
//                                                                                                                                
#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FastSimulation/Tracking/plugins/FastTrackingMaskProducer.h"
#include <vector>
#include <stdio.h>

FastTrackingMaskProducer::FastTrackingMaskProducer(const edm::ParameterSet& conf) 
  : oldHitMasks_exists_(false)
  , oldHitCombinationMasks_exists_(false)
  , overRideTrkQuals_(false)
  , filterTracks_(false)
{
  // Main products                                                                                                                 
  produces<std::vector<bool> >("hitMasks");
  produces<std::vector<bool> >("hitCombinationMasks");

  // Track collection                                                                                                                   
  edm::InputTag trackCollectionTag = conf.getParameter<edm::InputTag>("trackCollection");
  trackToken_ = consumes<reco::TrackCollection>(trackCollectionTag);

  // old hit masks
  oldHitMasks_exists_ = conf.exists("oldHitMasks");
  if (oldHitMasks_exists_){
    edm::InputTag hitMasksTag = conf.getParameter<edm::InputTag>("oldHitMasks");
    hitMasksToken_ = consumes<std::vector<bool> >(hitMasksTag);
  }

  // old hit combination masks
  oldHitCombinationMasks_exists_ = conf.exists("oldHitCombinationMasks");
  if (  oldHitCombinationMasks_exists_){
    edm::InputTag hitCombinationMasksTag = conf.getParameter<edm::InputTag>("oldHitCombinationMasks");
    hitCombinationMasksToken_ = consumes<std::vector<bool> >(hitCombinationMasksTag);
  }
  
  // read track quality from value map rather than from track itself
  overRideTrkQuals_ = conf.exists("overrideTrkQuals");
  if(  overRideTrkQuals_ ){
    edm::InputTag trkQualsTag = conf.getParameter<edm::InputTag>("overrideTrkQuals");
    if(trkQualsTag == edm::InputTag(""))
      overRideTrkQuals_ = false;
    else
      trkQualsToken_ = consumes<edm::ValueMap<int> >(trkQualsTag);
  }

  // required track quality
  trackQuality_=reco::TrackBase::undefQuality;
  if (conf.exists("TrackQuality")){
    filterTracks_=true;
    std::string trackQualityStr = conf.getParameter<std::string>("TrackQuality");
    if ( !trackQualityStr.empty() ) {
      trackQuality_=reco::TrackBase::qualityByName(trackQualityStr);
    }
  }
}

void
FastTrackingMaskProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // the products
  std::auto_ptr<std::vector<bool> > hitMasks(new std::vector<bool>());
  std::auto_ptr<std::vector<bool> > hitCombinationMasks(new std::vector<bool>());

  // The input track collection handle                                                                                           
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(trackToken_,trackCollection);

  // the track quality collection
  edm::Handle<edm::ValueMap<int> > quals;
  if ( overRideTrkQuals_ ) {
    e.getByToken(trkQualsToken_,quals);
  }

  // The input hitMasks handle
  if (oldHitMasks_exists_ == true){
    edm::Handle<std::vector<bool> > oldHitMasks;
    e.getByToken(hitMasksToken_,oldHitMasks);
    hitMasks->insert(hitMasks->begin(),oldHitMasks->begin(),oldHitMasks->end());
  }

  // The input hitCombinationMasks handle
  if (oldHitCombinationMasks_exists_ == true){
    edm::Handle<std::vector<bool> > oldHitCombinationMasks;
    e.getByToken(hitCombinationMasksToken_,oldHitCombinationMasks);
    hitCombinationMasks->insert(hitCombinationMasks->begin(),oldHitCombinationMasks->begin(),oldHitCombinationMasks->end());
 }

  int ngood = 0;
  //std::cout << "FTMP 1 " << trackCollection->size()<< std::endl;
  for (size_t i = 0 ; i!=trackCollection->size();++i)
    {
      

      const reco::Track & track = trackCollection->at(i);
      reco::TrackRef trackRef(trackCollection,i);
      if (filterTracks_) {
	bool goodTk = true;

	if ( overRideTrkQuals_ ) {
	  int qual= (*quals)[trackRef];
	  if ( qual < 0 ){
	    goodTk=false;
	  }
	  else
	    goodTk = ( qual & (1<<trackQuality_))>>trackQuality_;
	}
	else {
	  goodTk=(track.quality(trackQuality_));
	}
	if ( !goodTk) continue;
      }
      ngood++;
      //std::cout << "FTMP 2 " << ngood << std::endl;
      
      
      // Loop over the recHits
      // todo: implement the minimum number of measurements criterium
      // see http://cmslxr.fnal.gov/lxr/source/RecoLocalTracker/SubCollectionProducers/src/TrackClusterRemover.cc#0166
      for (auto hitIt = track.recHitsBegin() ;  hitIt != track.recHitsEnd(); ++hitIt) {

	if(!(*hitIt)->isValid())
	  continue;

	const GSSiTrackerRecHit2DLocalPos * hit = dynamic_cast<const GSSiTrackerRecHit2DLocalPos*>(*hitIt);
	if(hit){
	  //std::cout << "FTMP: I'm here" << " " << hit->hitCombinationId() << std::endl;
	  uint32_t hitCombination_id = hit->hitCombinationId();
	  if (hitCombination_id >= hitCombinationMasks->size()) { 
	    hitCombinationMasks->resize(hitCombination_id+1,false);
	  }
	  //std::cout << "hc " << hit->hitCombinationId() <<  " " << hitCombination_id << " " <<  hitCombinationMasks->size() << std::endl;
	  hitCombinationMasks->at(hitCombination_id) = true;
	  
	  /* hit id not yet properly implemented
	  uint32_t hit_id = hit->id();	
	  if (hit_id >= hitMasks->size()) { 
	    hitMasks->resize(hit_id+1,false);   
	  }
	  std::cout <<  "h " << hit->id() << " " << hit_id << " " <<  hitMasks->size() << std::endl;
	  hitMasks->at(hit_id) = true;
	  */
	}
	
	else{
	  continue;
	  // TODO: find out why the cast doesn't work every so many hits
	}
      }
    }

  //std::cout << "FTMP: 3 " <<  hitCombinationMasks->size() << std::endl;
  e.put(hitMasks,"hitMasks");
  e.put(hitCombinationMasks,"hitCombinationMasks");
}

