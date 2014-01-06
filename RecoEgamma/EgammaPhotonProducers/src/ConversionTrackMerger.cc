//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           ConversionTrackMerger
// 
// Description:     Merger for ConversionTracks, adapted from SimpleTrackListMerger
//
// Original Author: J.Bendavid
//
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

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

    
  ConversionTrackMerger::ConversionTrackMerger(edm::ParameterSet const& conf) : 
    conf_(conf)
  {
    // retrieve producer name of input TrackCollection(s)
    trackProducer1 = consumes<reco::ConversionTrackCollection>(conf_.getParameter<edm::InputTag>("TrackProducer1"));
    trackProducer2 = consumes<reco::ConversionTrackCollection>(conf_.getParameter<edm::InputTag>("TrackProducer2"));

    produces<reco::ConversionTrackCollection>();
   
  }


  // Virtual destructor needed.
  ConversionTrackMerger::~ConversionTrackMerger() { }  

  // Functions that gets called by framework every event
  void ConversionTrackMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    

    double shareFrac =  conf_.getParameter<double>("ShareFrac");
    bool allowFirstHitShare = conf_.getParameter<bool>("allowFirstHitShare");
    bool checkCharge = conf_.getParameter<bool>("checkCharge");    
    double minProb = conf_.getParameter<double>("minProb");
    
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
    e.getByToken(trackProducer1, trackCollection1);
    if (trackCollection1.isValid()) {
      TC1 = trackCollection1.product();
      //std::cout << "1st collection " << trackProducer1 << " has "<< TC1->size() << " tracks" << std::endl ;
    } else {
      TC1 = &s_empty1;
      edm::LogWarning("ConversionTrackMerger") 
	<< "1st TrackCollection not found;"
	<< " will only clean 2nd TrackCollection ";
    }
    reco::ConversionTrackCollection tC1 = *TC1;

    const reco::ConversionTrackCollection *TC2 = 0;
    edm::Handle<reco::ConversionTrackCollection> trackCollection2;
    e.getByToken(trackProducer2, trackCollection2);
    if (trackCollection2.isValid()) {
      TC2 = trackCollection2.product();
      //std::cout << "2nd collection " << trackProducer2 << " has "<< TC2->size() << " tracks" << std::endl ;
    } else {
        TC2 = &s_empty2;
        edm::LogWarning("ConversionTrackMerger") 
	  << "2nd TrackCollection not found;"
	  <<" will only clean 1st TrackCollection ";
    }
    reco::ConversionTrackCollection tC2 = *TC2;

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
    for (reco::ConversionTrackCollection::iterator track=tC1.begin(); track!=tC1.end(); ++track){
      i++; 
        
      //clear flags if preferCollection was set to 0
      selected1[i] = selected1[i] && outputPreferCollection!=0;
      track->setIsTrackerOnly ( track->isTrackerOnly() &&  trackerOnlyPreferCollection!=0 );
      track->setIsArbitratedEcalSeeded( track->isArbitratedEcalSeeded() &&  arbitratedEcalSeededPreferCollection!=0 );
      track->setIsArbitratedMerged( track->isArbitratedMerged() && arbitratedMergedPreferCollection!=0 );
      track->setIsArbitratedMergedEcalGeneral( track->isArbitratedMergedEcalGeneral() && arbitratedMergedEcalGeneralPreferCollection!=0 );
      

      std::vector<const TrackingRecHit*>& iHits = rh1[track]; 
      unsigned nh1 = iHits.size();
      int j=-1;
      for (reco::ConversionTrackCollection::iterator track2=tC2.begin(); track2!=tC2.end(); ++track2){
        j++;

        //clear flags if preferCollection was set to 0
        selected2[j] = selected2[j] && outputPreferCollection!=0;
        track2->setIsTrackerOnly ( track2->isTrackerOnly() &&  trackerOnlyPreferCollection!=0 );
        track2->setIsArbitratedEcalSeeded( track2->isArbitratedEcalSeeded() &&  arbitratedEcalSeededPreferCollection!=0 );
        track2->setIsArbitratedMerged( track2->isArbitratedMerged() && arbitratedMergedPreferCollection!=0 );
        track2->setIsArbitratedMergedEcalGeneral( track2->isArbitratedMergedEcalGeneral() && arbitratedMergedEcalGeneralPreferCollection!=0 );

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
        //if (noverlap>0) printf("noverlap = %i, firstoverlap = %i, nhit1 = %i, nhit2 = %i, algo1 = %i, algo2 = %i, q1 = %i, q2 = %i\n",noverlap,firstoverlap,nhit1,nhit2,track->track()->algo(),track2->track()->algo(),track->track()->charge(),track2->track()->charge());
	//std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->numberOfValidHits() << " "  << track2->numberOfValidHits() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ( (noverlap-firstoverlap) > (std::min(nhit1,nhit2)-firstoverlap)*shareFrac && (!checkCharge || track->track()->charge()*track2->track()->charge()>0) ) {
          //printf("overlapping tracks\n");
         //printf ("ndof1 = %5f, chisq1 = %5f, ndof2 = %5f, chisq2 = %5f\n",track->track()->ndof(),track->track()->chi2(),track2->track()->ndof(),track2->track()->chi2());
          
          double probFirst = ChiSquaredProbability(track->track()->chi2(), track->track()->ndof());
          double probSecond = ChiSquaredProbability(track2->track()->chi2(), track2->track()->ndof());

          //arbitrate by number of hits and reduced chisq
          bool keepFirst = ( nhit1>nhit2 || (nhit1==nhit2 && track->track()->normalizedChi2()<track2->track()->normalizedChi2()) );

          //override decision in case one track is radically worse quality than the other
          keepFirst |= (probFirst>minProb && probSecond<=minProb);
          keepFirst &= !(probFirst<=minProb && probSecond>minProb);

          bool keepSecond = !keepFirst;
                    
          //set flags based on arbitration decision and precedence settings

          selected1[i] =            selected1[i]            && ( (keepFirst && outputPreferCollection==3) || outputPreferCollection==-1 || outputPreferCollection==1 );
          track->setIsTrackerOnly ( track->isTrackerOnly() && ( (keepFirst && trackerOnlyPreferCollection==3) || trackerOnlyPreferCollection==-1 || trackerOnlyPreferCollection==1 ) );
          track->setIsArbitratedEcalSeeded( track->isArbitratedEcalSeeded() &&  ( (keepFirst && arbitratedEcalSeededPreferCollection==3) || arbitratedEcalSeededPreferCollection==-1 || arbitratedEcalSeededPreferCollection==1 ) );
          track->setIsArbitratedMerged( track->isArbitratedMerged() && ( (keepFirst && arbitratedMergedPreferCollection==3) || arbitratedMergedPreferCollection==-1 || arbitratedMergedPreferCollection==1 ) );
          track->setIsArbitratedMergedEcalGeneral( track->isArbitratedMergedEcalGeneral() && ( (keepFirst && arbitratedMergedEcalGeneralPreferCollection==3) || arbitratedMergedEcalGeneralPreferCollection==-1 || arbitratedMergedEcalGeneralPreferCollection==1 ) );
          
          selected2[j] =             selected2[j]            && ( (keepSecond && outputPreferCollection==3) || outputPreferCollection==-1 || outputPreferCollection==2 );
          track2->setIsTrackerOnly ( track2->isTrackerOnly() && ( (keepSecond && trackerOnlyPreferCollection==3) || trackerOnlyPreferCollection==-1 || trackerOnlyPreferCollection==2 ) );
          track2->setIsArbitratedEcalSeeded( track2->isArbitratedEcalSeeded() &&  ( (keepSecond && arbitratedEcalSeededPreferCollection==3) || arbitratedEcalSeededPreferCollection==-1 || arbitratedEcalSeededPreferCollection==2 ) );
          track2->setIsArbitratedMerged( track2->isArbitratedMerged() && ( (keepSecond && arbitratedMergedPreferCollection==3) || arbitratedMergedPreferCollection==-1 || arbitratedMergedPreferCollection==2 ) );
          track2->setIsArbitratedMergedEcalGeneral( track2->isArbitratedMergedEcalGeneral() && ( (keepSecond && arbitratedMergedEcalGeneralPreferCollection==3) || arbitratedMergedEcalGeneralPreferCollection==-1 || arbitratedMergedEcalGeneralPreferCollection==2 ) );
          
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
      if (!selected1[i]){
	continue;
      }
      //fill the TrackCollection
      outputTrks->push_back(*track);      
    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 1st collection
 
   
   if ( 0<tC2.size() ){
    i=0;
    for (reco::ConversionTrackCollection::const_iterator track=tC2.begin(); track!=tC2.end();
	 ++track, ++i){
      //don't store tracks rejected as duplicates
      if (!selected2[i]){
        continue;
      }
      //fill the TrackCollection
      outputTrks->push_back(*track);
    }//end faux loop over tracks
   }//end more than 0 track
  
    e.put(outputTrks);
    return;

  }//end produce
