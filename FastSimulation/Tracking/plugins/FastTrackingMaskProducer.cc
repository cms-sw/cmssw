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
{
  //Main products                                                                                                                 
  produces<std::vector<bool> >("hitMasks");
  produces<std::vector<bool> >("hitCombinationMasks");

  // Input Tag                                                                                                                   
  edm::InputTag trackCollectionTag = conf.getParameter<edm::InputTag>("trackCollection");

  oldHitMasks_exist = conf.exists("oldHitMasks");
  oldHitCombinationMasks_exist = conf.exists("oldHitCombinationMasks");

  if (oldHitMasks_exist){
    edm::InputTag hitMasksTag = conf.getParameter<edm::InputTag>("oldHitMasks");
    hitMasksToken = consumes<std::vector<bool> >(hitMasksTag);
  }

  if (  oldHitCombinationMasks_exist){
    edm::InputTag hitCombinationMasksTag = conf.getParameter<edm::InputTag>("oldHitCombinationMasks");
    hitCombinationMasksToken = consumes<std::vector<bool> >(hitCombinationMasksTag);
  }
  
  if (conf.exists("overrideTrkQuals")) {
    edm::InputTag overrideTrkQuals = conf.getParameter<edm::InputTag>("overrideTrkQuals");
    if ( !(overrideTrkQuals==edm::InputTag("")) )
      overrideTrkQuals_.push_back( consumes<edm::ValueMap<int> >(overrideTrkQuals) );
  }
  trackQuality_=reco::TrackBase::undefQuality;
  filterTracks_=false;
  if (conf.exists("TrackQuality")){
    filterTracks_=true;
    std::string trackQuality = conf.getParameter<std::string>("TrackQuality");
    if ( !trackQuality.empty() ) {
      trackQuality_=reco::TrackBase::qualityByName(trackQuality);
    }
  }

  // consumes                                                                                                                    
  trackToken = consumes<reco::TrackCollection>(trackCollectionTag);
}

void
FastTrackingMaskProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // The input track collection handle                                                                                           
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(trackToken,trackCollection);

  std::vector<edm::Handle<edm::ValueMap<int> > > quals;
  if ( overrideTrkQuals_.size() > 0) {
    quals.resize(1);
    e.getByToken(overrideTrkQuals_[0],quals[0]);
  }

  std::auto_ptr<std::vector<bool> > hitMasks(new std::vector<bool>());
  std::auto_ptr<std::vector<bool> > hitCombinationMasks(new std::vector<bool>());

  // The input hitMasks handle
  if (oldHitMasks_exist == true){
    edm::Handle<std::vector<bool> > oldHitMasks;
    e.getByToken(hitMasksToken,oldHitMasks);
    hitMasks->insert(hitMasks->begin(),oldHitMasks->begin(),oldHitMasks->end());
  }

  // The input hitCombinationMasks handle
  if (oldHitCombinationMasks_exist == true){
    edm::Handle<std::vector<bool> > oldHitCombinationMasks;
    e.getByToken(hitCombinationMasksToken,oldHitCombinationMasks);   // NOTE: in the 2nd iteration there is no 'oldhitmasks'
    hitCombinationMasks->insert(hitCombinationMasks->begin(),oldHitCombinationMasks->begin(),oldHitCombinationMasks->end());
 }

  for (size_t i = 0 ; i!=trackCollection->size();++i)
    {

      const reco::Track & track = trackCollection->at(i);
      reco::TrackRef trackRef(trackCollection,i);
      if (filterTracks_) {
	bool goodTk = true;

	if ( quals.size()!=0) {
	  int qual=(*(quals[0]))[trackRef];
	  //std::cout << qual << std::endl;                                                                                         
	  if ( qual < 0 ) {goodTk=false;}
	  //note that this does not work for some trackquals (goodIterative or undefQuality)                                        
	  else
	    goodTk = ( qual & (1<<trackQuality_))>>trackQuality_;
	}
	else
	  goodTk=(track.quality(trackQuality_));
	if ( !goodTk) continue;
      }
 

      // Loop over the recHits                                                                     
      for (auto hitIt = track.recHitsBegin() ;  hitIt != track.recHitsEnd(); ++hitIt) {

	const SiTrackerGSMatchedRecHit2D* hit = dynamic_cast<const SiTrackerGSMatchedRecHit2D*>(*hitIt);
	if (hit == 0){
	  std::cout << "Error in FastTrackingMaskProducer: dynamic hit cast failed" << std::endl;
	}

	if (hit->id() >= abs(hitMasks->size()+1)) {
	  hitMasks->resize(hit->id()+1,false);   
	}

	if (hit->id() >= abs(hitCombinationMasks->size()+1)) {
	  hitCombinationMasks->resize(hit->id()+1,false);   
	}

	hitMasks->at(hit->id()) = true;
	hitCombinationMasks->at(hit->hitCombinationId()) = true;
      }
    }
  e.put(hitMasks,"hitMasks");
  e.put(hitCombinationMasks,"hitCombinationMasks");
}
