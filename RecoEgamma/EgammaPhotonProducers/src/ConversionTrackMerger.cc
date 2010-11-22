//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackMerger
// 
// Description:     Merger for ConversionTracks, adapted from SimpleTrackListMerger
//
// Original Author: J.Bendavid
//
// $Author: bendavid $
// $Date: 2010/09/17 19:46:18 $
// $Revision: 1.1 $
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackMerger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "DataFormats/TrackReco/src/classes.h"

#include "RecoTracker/TrackProducer/interface/ClusterRemovalRefSetter.h"
    
  ConversionTrackMerger::ConversionTrackMerger(edm::ParameterSet const& conf) : 
    conf_(conf)
  {
    produces<reco::ConversionTrackCollection>();
   
  }


  // Virtual destructor needed.
  ConversionTrackMerger::~ConversionTrackMerger() { }  

  // Functions that gets called by framework every event
  void ConversionTrackMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input TrackCollection(s)
    std::string trackProducer1 = conf_.getParameter<std::string>("TrackProducer1");
    std::string trackProducer2 = conf_.getParameter<std::string>("TrackProducer2");

    double shareFrac =  conf_.getParameter<double>("ShareFrac");
    bool allowFirstHitShare = conf_.getParameter<bool>("allowFirstHitShare");
    
    int outputPreferCollection = conf_.getParameter<int>("outputPreferCollection");
    int trackerOnlyPreferCollection = conf_.getParameter<int>("trackerOnlyPreferCollection");
    int arbitratedEcalSeededPreferCollection = conf_.getParameter<int>("arbitratedEcalSeededPreferCollection");    
    int arbitratedMergedPreferCollection = conf_.getParameter<int>("arbitratedMergedPreferCollection");
    int arbitratedMergedEcalGeneralPreferCollection = conf_.getParameter<int>("arbitratedMergedEcalGeneralPreferCollection");    

    // get Inputs 
    // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
    // this allows ConversionTrackMerger to be used as a cleaner only if handed just one list
    // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
    //
    const reco::ConversionTrackCollection *TC1 = 0;
    static const reco::ConversionTrackCollection s_empty1, s_empty2;
    edm::Handle<reco::ConversionTrackCollection> trackCollection1;
    e.getByLabel(trackProducer1, trackCollection1);
    if (trackCollection1.isValid()) {
      TC1 = trackCollection1.product();
      //std::cout << "1st collection " << trackProducer1 << " has "<< TC1->size() << " tracks" << std::endl ;
    } else {
      TC1 = &s_empty1;
      edm::LogWarning("ConversionTrackMerger") << "1st TrackCollection " << trackProducer1 << " not found; will only clean 2nd TrackCollection " << trackProducer2 ;
    }
    const reco::ConversionTrackCollection tC1 = *TC1;

    const reco::ConversionTrackCollection *TC2 = 0;
    edm::Handle<reco::ConversionTrackCollection> trackCollection2;
    e.getByLabel(trackProducer2, trackCollection2);
    if (trackCollection2.isValid()) {
      TC2 = trackCollection2.product();
      //std::cout << "2nd collection " << trackProducer2 << " has "<< TC2->size() << " tracks" << std::endl ;
    } else {
        TC2 = &s_empty2;
        edm::LogWarning("ConversionTrackMerger") << "2nd TrackCollection " << trackProducer2 << " not found; will only clean 1st TrackCollection " << trackProducer1 ;
    }
    const reco::ConversionTrackCollection tC2 = *TC2;

    // Step B: create empty output collection
    outputTrks = std::auto_ptr<reco::ConversionTrackCollection>(new reco::ConversionTrackCollection);
    int i;

    std::vector<int> selected1; for (unsigned int i=0; i<tC1.size(); ++i){selected1.push_back(1);}
    std::vector<int> selected2; for (unsigned int i=0; i<tC2.size(); ++i){selected2.push_back(1);}

   
   std::map<reco::ConversionTrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh1;
   std::map<reco::ConversionTrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh2;
   for (reco::ConversionTrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
     trackingRecHit_iterator itB = track->track()->recHitsBegin();
     trackingRecHit_iterator itE = track->track()->recHitsEnd();
     for (trackingRecHit_iterator it = itB;  it != itE; ++it) { 
       const TrackingRecHit* hit = &(**it);
       rh1[track].push_back(hit);
     }
   }
   for (reco::ConversionTrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); ++track){
     trackingRecHit_iterator jtB = track->track()->recHitsBegin();
     trackingRecHit_iterator jtE = track->track()->recHitsEnd();
     for (trackingRecHit_iterator jt = jtB;  jt != jtE; ++jt) { 
       const TrackingRecHit* hit = &(**jt);
       rh2[track].push_back(hit);
     }
   }

   if ( (0<tC1.size())&&(0<tC2.size()) ){
    i=-1;
    for (reco::ConversionTrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
      i++; 
      std::vector<const TrackingRecHit*>& iHits = rh1[track]; 
      unsigned nh1 = iHits.size();
      int j=-1;
      for (reco::ConversionTrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); ++track2){
        j++;
	std::vector<const TrackingRecHit*>& jHits = rh2[track2]; 
	unsigned nh2 = jHits.size();
        int noverlap=0;
        int firstoverlap=0;
	for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  const TrackingRecHit* it = iHits[ih];
          if (it->isValid()){
            int jj=-1;
	    for ( unsigned jh=0; jh<nh2; ++jh ) { 
	      const TrackingRecHit* jt = jHits[jh];
              jj++;
	      if (jt->isValid()){           
		if ( it->sharesInput(jt,TrackingRecHit::some) ) {
		  noverlap++;
		  if ( allowFirstHitShare && ( ih == 0 ) && ( jh == 0 ) ) firstoverlap=1;
		}
	      }
            }
          }
        }
	int nhit1 = track->track()->numberOfValidHits();
	int nhit2 = track2->track()->numberOfValidHits();
	//std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->numberOfValidHits() << " "  << track2->numberOfValidHits() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ( (noverlap-firstoverlap) > (std::min(nhit1,nhit2)-firstoverlap)*shareFrac ) {
          if ( nhit1 > nhit2 ){
            selected2[j]=0; 
	    //std::cout << " removing L2 trk in pair " << std::endl;
          }else{
            if ( nhit1 < nhit2 ){
              selected1[i]=0; 
	      //std::cout << " removing L1 trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << track->normalizedChi2() << " " << track2->normalizedChi2() << std::endl;
              if (track->track()->normalizedChi2() > track2->track()->normalizedChi2()) {
		selected1[i]=0;
	      }else {
		selected2[j]=0;
              }
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track

  //
  //  output selected tracks - if any
  //
   
   if ( 0<tC1.size() ){
     i=0;
     for (reco::ConversionTrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); 
	  ++track, ++i){
      //don't store tracks rejected as duplicates
      if ( outputPreferCollection==0 || ( (outputPreferCollection==3 || outputPreferCollection==2) && !selected1[i]) ){
	continue;
      }
      //fill the TrackCollection
      outputTrks->push_back(*track);
      //clear flags for tracks rejected as duplicates
      if ( trackerOnlyPreferCollection==0 || ( (trackerOnlyPreferCollection==3 || trackerOnlyPreferCollection==2) && !selected1[i]) ){
        outputTrks->back().setIsTrackerOnly(false);
      }
      if ( arbitratedEcalSeededPreferCollection==0 || ( (arbitratedEcalSeededPreferCollection==3 || arbitratedEcalSeededPreferCollection==2) && !selected1[i]) ){
        outputTrks->back().setIsArbitratedEcalSeeded(false);
      }      
      if ( arbitratedMergedPreferCollection==0 || ( (arbitratedMergedPreferCollection==3 || arbitratedMergedPreferCollection==2) && !selected1[i]) ){
        outputTrks->back().setIsArbitratedMerged(false);
      }    
      if ( arbitratedMergedEcalGeneralPreferCollection==0 || ( (arbitratedMergedEcalGeneralPreferCollection==3 || arbitratedMergedEcalGeneralPreferCollection==2) && !selected1[i]) ){
        outputTrks->back().setIsArbitratedMergedEcalGeneral(false);
      }       
    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 1st collection
 
   
   if ( 0<tC2.size() ){
    i=0;
    for (reco::ConversionTrackCollection::const_iterator track=tC2.begin(); track!=tC2.end();
	 ++track, ++i){
      //don't store tracks rejected as duplicates
      if ( outputPreferCollection==0 || ( (outputPreferCollection==3 || outputPreferCollection==1) && !selected2[i]) ){
        continue;
      }
      //fill the TrackCollection
      outputTrks->push_back(*track);
      //clear flags for tracks rejected as duplicates
      if ( trackerOnlyPreferCollection==0 || ( (trackerOnlyPreferCollection==3 || trackerOnlyPreferCollection==1) && !selected2[i]) ){
        outputTrks->back().setIsTrackerOnly(false);
      }
      if ( arbitratedEcalSeededPreferCollection==0 || ( (arbitratedEcalSeededPreferCollection==3 || arbitratedEcalSeededPreferCollection==1) && !selected2[i]) ){
        outputTrks->back().setIsArbitratedEcalSeeded(false);
      }      
      if ( arbitratedMergedPreferCollection==0 || ( (arbitratedMergedPreferCollection==3 || arbitratedMergedPreferCollection==1) && !selected2[i]) ){
        outputTrks->back().setIsArbitratedMerged(false);
      }      
      if ( arbitratedMergedEcalGeneralPreferCollection==0 || ( (arbitratedMergedEcalGeneralPreferCollection==3 || arbitratedMergedEcalGeneralPreferCollection==1) && !selected2[i]) ){
        outputTrks->back().setIsArbitratedMergedEcalGeneral(false);
      }            

    }//end faux loop over tracks
   }//end more than 0 track
 
    e.put(outputTrks);
    return;

  }//end produce
