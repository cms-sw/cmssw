//
// Package:         RecoTracker/FinalTrackSelectors
// Class:           SimpleTrackListMerger
// 
// Description:     TrackList Cleaner and Merger
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: burkett $
// $Date: 2008/06/25 23:09:24 $
// $Revision: 1.15 $
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoTracker/FinalTrackSelectors/interface/SimpleTrackListMerger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "DataFormats/TrackReco/src/classes.h"

namespace cms
{

  SimpleTrackListMerger::SimpleTrackListMerger(edm::ParameterSet const& conf) : 
    conf_(conf)
  {

    produces<reco::TrackCollection>();
    produces<reco::TrackExtraCollection>();
    produces<TrackingRecHitCollection>();
    produces< std::vector<Trajectory> >();
    produces< TrajTrackAssociationCollection >();

  }


  // Virtual destructor needed.
  SimpleTrackListMerger::~SimpleTrackListMerger() { }  

  // Functions that gets called by framework every event
  void SimpleTrackListMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input TrackCollection(s)
    std::string trackProducer1 = conf_.getParameter<std::string>("TrackProducer1");
    std::string trackProducer2 = conf_.getParameter<std::string>("TrackProducer2");

    double maxNormalizedChisq =  conf_.getParameter<double>("MaxNormalizedChisq");
    double minPT =  conf_.getParameter<double>("MinPT");
    unsigned int minFound = (unsigned int)conf_.getParameter<int>("MinFound");
    double epsilon =  conf_.getParameter<double>("Epsilon");
    bool use_sharesInput = true;
    if ( epsilon > 0.0 )use_sharesInput = false;
    double shareFrac =  conf_.getParameter<double>("ShareFrac");
  
    bool promoteQuality = conf_.getParameter<bool>("promoteTrackQuality");
//

    // New track quality should be read from the file
    std::string qualityStr = conf_.getParameter<std::string>("newQuality");
    reco::TrackBase::TrackQuality qualityToSet;
    if (qualityStr != "") {
      qualityToSet = reco::TrackBase::qualityByName(conf_.getParameter<std::string>("newQuality"));
    }
    else 
      qualityToSet = reco::TrackBase::undefQuality;

    // extract tracker geometry
    //
    edm::ESHandle<TrackerGeometry> theG;
    es.get<TrackerDigiGeometryRecord>().get(theG);

//    using namespace reco;

    // get Inputs 
    // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
    // this allows SimpleTrackListMerger to be used as a cleaner only if handed just one list
    // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
    //
    const reco::TrackCollection *TC1 = 0;
    static const reco::TrackCollection s_empty1, s_empty2;
    edm::Handle<reco::TrackCollection> trackCollection1;
    e.getByLabel(trackProducer1, trackCollection1);
    if (trackCollection1.isValid()) {
      TC1 = trackCollection1.product();
      //std::cout << "1st collection " << trackProducer1 << " has "<< TC1->size() << " tracks" << std::endl ;
    } else {
      TC1 = &s_empty1;
      edm::LogWarning("SimpleTrackListMerger") << "1st TrackCollection " << trackProducer1 << " not found; will only clean 2nd TrackCollection " << trackProducer2 ;
    }
    const reco::TrackCollection tC1 = *TC1;

    const reco::TrackCollection *TC2 = 0;
    edm::Handle<reco::TrackCollection> trackCollection2;
    e.getByLabel(trackProducer2, trackCollection2);
    if (trackCollection2.isValid()) {
      TC2 = trackCollection2.product();
      //std::cout << "2nd collection " << trackProducer2 << " has "<< TC2->size() << " tracks" << std::endl ;
    } else {
        TC2 = &s_empty2;
        edm::LogWarning("SimpleTrackListMerger") << "2nd TrackCollection " << trackProducer2 << " not found; will only clean 1st TrackCollection " << trackProducer1 ;
    }
    const reco::TrackCollection tC2 = *TC2;

    // Step B: create empty output collection
    outputTrks = std::auto_ptr<reco::TrackCollection>(new reco::TrackCollection);
    refTrks = e.getRefBeforePut<reco::TrackCollection>();      

    outputTrkExtras = std::auto_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection);
    refTrkExtras    = e.getRefBeforePut<reco::TrackExtraCollection>();
    outputTrkHits   = std::auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection);
    refTrkHits      = e.getRefBeforePut<TrackingRecHitCollection>();
    outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
    outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());

  //
  //  no input tracks
  //

//    if ( tC1.empty() ){
//      LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
//      e.put(output);
//      return;  
//    }

  //
  //  quality cuts first
  // 
    int i;

    std::vector<int> selected1; for (unsigned int i=0; i<tC1.size(); ++i){selected1.push_back(1);}

   if ( 0<tC1.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in SimpleTrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track


    std::vector<int> selected2; for (unsigned int i=0; i<tC2.size(); ++i){selected2.push_back(1);}

   if ( 0<tC2.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in SimpleTrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track

   //DON'T BELIEVE WE NEED THIS LOOP
   /*
  //
  //  L1 has > 1 track - try merging
  //
   if ( 1<tC1.size() ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
      i++; 
      //std::cout << "Track number "<< i << std::endl ; 
      if (!selected1[i])continue;
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC1.begin(); track2!=tC1.end(); track2++){
        j++;
        if ((j<=i)||(!selected1[j])||(!selected1[i]))continue;
        int noverlap=0;
        for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
          if ((*it)->isValid()){
            for (trackingRecHit_iterator jt = track2->recHitsBegin();  jt != track2->recHitsEnd(); jt++){
	      if ((*jt)->isValid()){
//                if (((*it)->geographicalId()==(*jt)->geographicalId())&&((*it)->localPosition().x()==(*jt)->localPosition().x()))noverlap++;
               if (!use_sharesInput){
                float delta = fabs ( (*it)->localPosition().x()-(*jt)->localPosition().x() ); 
                if (((*it)->geographicalId()==(*jt)->geographicalId())&&(delta<epsilon))noverlap++;
               }else{
                const TrackingRecHit* kt = &(**jt);
                if ( (*it)->sharesInput(kt,TrackingRecHit::some) )noverlap++;
               }
              }
            }
          }
        }
        float fi=float(noverlap)/float(track->numberOfValidHits()); float fj=float(noverlap)/float(track2->numberOfValidHits());
        //std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->numberOfValidHits() << " "  << track2->numberOfValidHits() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected1[j]=0;
            //std::cout << " removing 2nd trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected1[i]=0; 
              //std::cout << " removing 1st trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()){selected1[i]=0;}else{selected1[j]=0;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track
   */

   //DON'T BELIEVE WE NEED THIS LOOP EITHER
   /*
  //
  //  L2 has > 1 track - try merging
  //
   if ( 1<tC2.size() ){
    int i=-1;
    for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
      i++; 
      //std::cout << "Track number "<< i << std::endl ; 
      if (!selected2[i])continue;
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); track2++){
        j++;
        if ((j<=i)||(!selected2[j])||(!selected2[i]))continue;
        int noverlap=0;
        for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
          if ((*it)->isValid()){
            for (trackingRecHit_iterator jt = track2->recHitsBegin();  jt != track2->recHitsEnd(); jt++){
	      if ((*jt)->isValid()){
//                if (((*it)->geographicalId()==(*jt)->geographicalId())&&((*it)->localPosition().x()==(*jt)->localPosition().x()))noverlap++;
               if (!use_sharesInput){
                float delta = fabs ( (*it)->localPosition().x()-(*jt)->localPosition().x() ); 
                if (((*it)->geographicalId()==(*jt)->geographicalId())&&(delta<epsilon))noverlap++;
               }else{
                const TrackingRecHit* kt = &(**jt);
                if ( (*it)->sharesInput(kt,TrackingRecHit::some) )noverlap++;
               }
              }
            }
          }
        }
        float fi=float(noverlap)/float(track->numberOfValidHits()); float fj=float(noverlap)/float(track2->numberOfValidHits());
        //std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->numberOfValidHits() << " "  << track2->numberOfValidHits() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected2[j]=0; 
            //std::cout << " removing 2nd trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected2[i]=0; 
              //std::cout << " removing 1st trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()){selected2[i]=0;}else{selected2[j]=0;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track
   */

  //
  //  L1 and L2 both have > 0 track - try merging
  //
   std::map<reco::TrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh1;
   std::map<reco::TrackCollection::const_iterator, std::vector<const TrackingRecHit*> > rh2;
   for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
     trackingRecHit_iterator itB = track->recHitsBegin();
     trackingRecHit_iterator itE = track->recHitsEnd();
     for (trackingRecHit_iterator it = itB;  it != itE; ++it) { 
       const TrackingRecHit* hit = &(**it);
       rh1[track].push_back(hit);
     }
   }
   for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); ++track){
     trackingRecHit_iterator jtB = track->recHitsBegin();
     trackingRecHit_iterator jtE = track->recHitsEnd();
     for (trackingRecHit_iterator jt = jtB;  jt != jtE; ++jt) { 
       const TrackingRecHit* hit = &(**jt);
       rh2[track].push_back(hit);
     }
   }

   if ( (0<tC1.size())&&(0<tC2.size()) ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); ++track){
      i++; 
      if (!selected1[i])continue;
      std::vector<const TrackingRecHit*>& iHits = rh1[track]; 
      unsigned nh1 = iHits.size();
      int qualityMaskT1 = track->qualityMask();
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); ++track2){
        j++;
        if ((!selected2[j])||(!selected1[i]))continue;
	std::vector<const TrackingRecHit*>& jHits = rh2[track2]; 
	unsigned nh2 = jHits.size();
        int noverlap=0;
	for ( unsigned ih=0; ih<nh1; ++ih ) { 
	  const TrackingRecHit* it = iHits[ih];
          if (it->isValid()){
            int jj=-1;
	    for ( unsigned jh=0; jh<nh2; ++jh ) { 
	      const TrackingRecHit* jt = jHits[jh];
              jj++;
	      if (jt->isValid()){
               if (!use_sharesInput){
                float delta = fabs ( it->localPosition().x()-jt->localPosition().x() ); 
                if ((it->geographicalId()==jt->geographicalId())&&(delta<epsilon))noverlap++;
               }else{
                const TrackingRecHit* kt = jt;
                if ( it->sharesInput(kt,TrackingRecHit::some) )noverlap++;
               }
              }
            }
          }
        }
	int newQualityMask = (qualityMaskT1 | track2->qualityMask()); // take OR of trackQuality 
        float fi=float(noverlap)/float(track->numberOfValidHits()); float fj=float(noverlap)/float(track2->numberOfValidHits());
	//std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->numberOfValidHits() << " "  << track2->numberOfValidHits() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected2[j]=0; 
	    selected1[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
            //std::cout << " removing L2 trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected1[i]=0; 
	      selected2[j]=10+newQualityMask;  // add 10 to avoid the case where mask = 1
              //std::cout << " removing L1 trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << track->normalizedChi2() << " " << track2->normalizedChi2() << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()) {
		selected1[i]=0;
		selected2[j]=10+newQualityMask; // add 10 to avoid the case where mask = 1
	      }else{
		selected2[j]=0;
		selected1[i]=10+newQualityMask; // add 10 to avoid the case where mask = 1
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
   trackRefs.resize(tC1.size()+tC2.size());
   size_t current = 0;

   if ( 0<tC1.size() ){
     i=0;
     for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); 
	  ++track, ++current, ++i){
      if (!selected1[i]){
	trackRefs[current] = reco::TrackRef();
	continue;
      }
      const reco::Track & theTrack = * track;
      //fill the TrackCollection
      outputTrks->push_back( reco::Track( theTrack ) );
      if (selected1[i]>1 && promoteQuality){
	outputTrks->back().setQualityMask(selected1[i]-10);
	outputTrks->back().setQuality(qualityToSet);
      }
      // Fill TrackExtra collection
      outputTrkExtras->push_back( reco::TrackExtra( 
		    theTrack.outerPosition(), theTrack.outerMomentum(), theTrack.outerOk(),
                    theTrack.innerPosition(), theTrack.innerMomentum(), theTrack.innerOk(),
                    theTrack.outerStateCovariance(), theTrack.outerDetId(),
                    theTrack.innerStateCovariance(), theTrack.innerDetId(),
                    theTrack.seedDirection(), theTrack.seedRef() ) );
      outputTrks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
      reco::TrackExtra & tx = outputTrkExtras->back();
      // fill TrackingRecHits
      std::vector<const TrackingRecHit*>& iHits = rh1[track]; 
      unsigned nh1 = iHits.size();
      for ( unsigned ih=0; ih<nh1; ++ih ) { 
	const TrackingRecHit* hit = iHits[ih];
	//for( trackingRecHit_iterator hit = itB; hit != itE; ++hit ) {
	outputTrkHits->push_back( hit->clone() );
	tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
      }
      trackRefs[current] = reco::TrackRef(refTrks, outputTrks->size() - 1);

    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 1st collection
   edm::Handle< std::vector<Trajectory> > hTraj1;
   e.getByLabel(trackProducer1, hTraj1);
   edm::Handle< TrajTrackAssociationCollection > hTTAss1;
   e.getByLabel(trackProducer1, hTTAss1);
 
   outputTrajs = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>()); 
   refTrajs    = e.getRefBeforePut< std::vector<Trajectory> >();
   outputTTAss = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());

   for (size_t i = 0, n = hTraj1->size(); i < n; ++i) {
     edm::Ref< std::vector<Trajectory> > trajRef(hTraj1, i);
     TrajTrackAssociationCollection::const_iterator match = hTTAss1->find(trajRef);
     if (match != hTTAss1->end()) {
       const edm::Ref<reco::TrackCollection> &trkRef = match->val; 
       short oldKey = static_cast<short>(trkRef.key());
       if (trackRefs[oldKey].isNonnull()) {
	 outputTrajs->push_back( Trajectory(*trajRef) );
	 outputTTAss->insert ( edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1), 
			       trackRefs[oldKey] );
       }
     }
   }

   short offset = current; //save offset into trackRefs

   if ( 0<tC2.size() ){
    i=0;
    for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end();
	 ++track, ++current, ++i){
      if (!selected2[i]){
	trackRefs[current] = reco::TrackRef();
	continue;
      }
      const reco::Track & theTrack = * track;
      //fill the TrackCollection
      outputTrks->push_back( reco::Track( theTrack ) );
      if (selected2[i]>1 && promoteQuality){
	outputTrks->back().setQualityMask(selected2[i]-10);
	outputTrks->back().setQuality(qualityToSet);
      }
      // Fill TrackExtra collection
      outputTrkExtras->push_back( reco::TrackExtra( 
		    theTrack.outerPosition(), theTrack.outerMomentum(), theTrack.outerOk(),
                    theTrack.innerPosition(), theTrack.innerMomentum(), theTrack.innerOk(),
                    theTrack.outerStateCovariance(), theTrack.outerDetId(),
                    theTrack.innerStateCovariance(), theTrack.innerDetId(),
                    theTrack.seedDirection(), theTrack.seedRef() ) );
      outputTrks->back().setExtra( reco::TrackExtraRef( refTrkExtras, outputTrkExtras->size() - 1) );
      reco::TrackExtra & tx = outputTrkExtras->back();
      // fill TrackingRecHits
      std::vector<const TrackingRecHit*>& jHits = rh2[track]; 
      unsigned nh2 = jHits.size();
      for ( unsigned jh=0; jh<nh2; ++jh ) { 
	const TrackingRecHit* hit = jHits[jh];
	outputTrkHits->push_back( hit->clone() );
	tx.add( TrackingRecHitRef( refTrkHits, outputTrkHits->size() - 1) );
      }
      trackRefs[current] = reco::TrackRef(refTrks, outputTrks->size() - 1);

    }//end faux loop over tracks
   }//end more than 0 track

   //Fill the trajectories, etc. for 2nd collection
   edm::Handle< std::vector<Trajectory> > hTraj2;
   e.getByLabel(trackProducer2, hTraj2);
   edm::Handle< TrajTrackAssociationCollection > hTTAss2;
   e.getByLabel(trackProducer2, hTTAss2);

   for (size_t i = 0, n = hTraj2->size(); i < n; ++i) {
     edm::Ref< std::vector<Trajectory> > trajRef(hTraj2, i);
     TrajTrackAssociationCollection::const_iterator match = hTTAss2->find(trajRef);
     if (match != hTTAss2->end()) {
       const edm::Ref<reco::TrackCollection> &trkRef = match->val; 
       short oldKey = static_cast<short>(trkRef.key()) + offset;
       if (trackRefs[oldKey].isNonnull()) {
	 outputTrajs->push_back( Trajectory(*trajRef) );
	 outputTTAss->insert ( edm::Ref< std::vector<Trajectory> >(refTrajs, outputTrajs->size() - 1), 
			       trackRefs[oldKey] ); 
       }
     }
   }

    e.put(outputTrks);
    e.put(outputTrkExtras);
    e.put(outputTrkHits);
    e.put(outputTrajs);
    e.put(outputTTAss);
    return;

  }//end produce
}
