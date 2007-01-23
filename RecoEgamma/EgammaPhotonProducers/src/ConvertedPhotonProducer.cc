#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonProducer.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
//
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theNavigationSchool_(0), 
  isInitialized(0)

{


   LogDebug("ConvertedPhotonProducer") << " CTOR " << "\n";

  // use onfiguration file to setup input collection names
 
  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<std::string>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<std::string>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");
  
  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  
  conversionOITrackProducer_ = conf_.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_ = conf_.getParameter<std::string>("conversionIOTrackProducer");


  // use onfiguration file to setup output collection names
  ConvertedPhotonCollection_     = conf_.getParameter<std::string>("convertedPhotonCollection");


  // Register the product
  produces< reco::ConvertedPhotonCollection >(ConvertedPhotonCollection_);


  // instantiate the Track Pair Finder algorithm
  theTrackPairFinder_ = new ConversionTrackPairFinder ();
  // instantiate the Vertex Finder algorithm
  theVertexFinder_ = new ConversionVertexFinder ();


}

ConvertedPhotonProducer::~ConvertedPhotonProducer() {


  delete theTrackPairFinder_;
  delete theVertexFinder_;

}


void  ConvertedPhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {

  // Inizilize my global event counter
  nEvt_=0;

  //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


  theEventSetup.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker_ );


  // get the measurement tracker   
  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  theEventSetup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  theMeasurementTracker_ = measurementTrackerHandle.product();
  
  theLayerMeasurements_  = new LayerMeasurements(theMeasurementTracker_);
  theNavigationSchool_   = new SimpleNavigationSchool( &(*theGeomSearchTracker_)  , &(*theMF_));
  NavigationSetter setter( *theNavigationSchool_);
  
  
}


void  ConvertedPhotonProducer::endJob () {

  edm::LogInfo("ConvertedPhotonProducer") << " Analyzed " << nEvt_  << "\n";
  LogDebug("ConvertedPhotonProducer") << "::endJob Analyzed " << nEvt_ << " events " << "\n";
 

}

void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;  
  edm::LogInfo("ConvertedPhotonProducer") << "Analyzing event number: " << theEvent.id() << " Global counter " << nEvt_  << "\n";
  LogDebug("ConvertedPhotonProducer") << "::produce event number " <<   theEvent.id() << " Global counter " << nEvt_ << "\n";
  
  
  //
  // create empty output collections
  //

  // Converted photon candidates
  reco::ConvertedPhotonCollection outputConvPhotonCollection;
  std::auto_ptr< reco::ConvertedPhotonCollection > outputConvPhotonCollection_p(new reco::ConvertedPhotonCollection);
  LogDebug("ConvertedPhotonProducer") << " Created empty ConvertedPhotonCollection size " <<   "\n";


  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  LogDebug("ConvertedPhotonProducer") << " Trying to access " << scHybridBarrelCollection_.c_str() << "  from my Producer " << "\n";
  
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  LogDebug("ConvertedPhotonProducer") << "barrel  SC collection size  " << scBarrelCollection.size() << "\n";
  
  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  LogDebug("ConvertedPhotonProducer") << " Trying to access " <<scIslandEndcapCollection_.c_str() << "  from my Producer " << "\n";
  
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  LogDebug("ConvertedPhotonProducer") << "Endcap SC collection size  " << scEndcapCollection.size() << "\n";

  //// Get the Out In CKF tracks from conversions
  Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  LogDebug("ConvertedPhotonProducer") << " outInTrack collection size " << (*outInTrkHandle).size() << "\n";

 // Loop over Out In Tracks
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkHandle).begin(); iTk !=  (*outInTrkHandle).end(); iTk++) {
    LogDebug("ConvertedPhotonProducer") << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
    
    LogDebug("ConvertedPhotonProducer") << " Out In Track Extra inner momentum  " << iTk->extra()->outerMomentum() << "\n";  
   
  }


  //// Get the In Out  CKF tracks from conversions
  Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  LogDebug("ConvertedPhotonProducer") << " inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";

  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder);
  //do the conversion:
  std::vector<reco::TransientTrack> t_outInTrk = ( *theTransientTrackBuilder ).build(outInTrkHandle );
  std::vector<reco::TransientTrack> t_inOutTrk = ( *theTransientTrackBuilder ).build(inOutTrkHandle );
  
  
  

  reco::ConvertedPhotonCollection myConvPhotons;

  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  int iSC=0; // index in photon collection
  int lSC=0; // local index on barrel
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {
    
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
    LogDebug("ConvertedPhotonProducer") << " SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";
    
    
    ///// Find the +/- pairs
    //std::vector<std::vector<reco::Track> > allPairs = theTrackPairFinder_->run(outInTrkHandle,  inOutTrkHandle );
    std::vector<std::vector<reco::TransientTrack> > allPairs = theTrackPairFinder_->run(t_outInTrk,  t_inOutTrk );
    
     LogDebug("ConvertedPhotonProducer") << " Barrel  allPairs.size " << allPairs.size() << "\n";
    edm::RefVector<reco::TrackCollection> trackPairRef; 
    // std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;



    if ( allPairs.size() ) {
      
      for ( std::vector<std::vector<reco::TransientTrack> >::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
	LogDebug("ConvertedPhotonProducer") << " Barrel single pair size " << (*iPair).size() << "\n";  
	if (  (*iPair).size()  < 2) continue;
	
	CachingVertex theConversionVertex=theVertexFinder_->run(*iPair);
        if ( theConversionVertex.isValid() ) {	
	  LogDebug("ConvertedPhotonProducer") << " conversion vertex position " << theConversionVertex.position() << "\n";
	} else {
	  LogDebug("ConvertedPhotonProducer") << " conversion vertex is not valid " << "\n";
	}
	
	//// Create a converted photon candidate per each track pair
	
        if (  theConversionVertex.isValid() ) {
	  const reco::Particle::Point  vtx( 0, 0, 0 );
	  //const reco::Particle::Point  convVtx(   theConversionVertex->position().x(), theConversionVertex->position().y(),  theConversionVertex->position().z() );
	  const reco::Particle::Point  convVtx(   theConversionVertex.position().x(), theConversionVertex.position().y(),  theConversionVertex.position().z() );
	   LogDebug("ConvertedPhotonProducer") << " SC energy " <<  aClus->energy() << "\n";
	  math::XYZVector direction =aClus->position() - vtx;
	  math::XYZVector momentum = direction.unit() * aClus->energy();
	  const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
	   LogDebug("ConvertedPhotonProducer") << " photon p4 " << p4  << "\n";
	  
	  
	  //// loop over tracks in the pair for creating a reference
	  trackPairRef.clear();
	  for ( std::vector<reco::TransientTrack>::const_iterator iTk=(*iPair).begin(); iTk!=(*iPair).end(); ++iTk) {
	    LogDebug("ConvertedPhotonProducer") << " Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
	    
	    reco::TrackRef myTkRef= iTk->persistentTrackRef(); 
	    LogDebug("ConvertedPhotonProducer") << " Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	    
	    
	    trackPairRef.push_back(myTkRef);
	    
	  }
	  
	  
	  LogDebug("ConvertedPhotonProducer") << " trackPairRef  " << trackPairRef.size() <<  "\n";
	  reco::ConvertedPhoton  newCandidate(0, p4, vtx, convVtx);
	  newCandidate.setP4(p4);
	  outputConvPhotonCollection.push_back(newCandidate);
	  // set the reference to the SC
	  reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
	  outputConvPhotonCollection[iSC].setSuperCluster(scRef);
	  // set the reference to the tracks
	  outputConvPhotonCollection[iSC].setTrackPairRef( trackPairRef);
	  
		  
	
	  iSC++;	
	  myCands++;
	  LogDebug("ConvertedPhotonProducer") << " Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	  
	  
	  // valid conversion vertex 
	} 
	
	
      }
      
    } else {
      
      LogDebug("ConvertedPhotonProducer") << " GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      
      const reco::Particle::Point  vtx( 0, 0, 0 );
      const reco::Particle::Point  convVtx(  0, 0, 0 );
      LogDebug("ConvertedPhotonProducer") << " SC energy " <<  aClus->energy() << "\n";
      math::XYZVector direction =aClus->position() - vtx;
      math::XYZVector momentum = direction.unit() * aClus->energy();
      const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
       LogDebug("ConvertedPhotonProducer") << " photon p4 " << p4  << "\n";
      
       LogDebug("ConvertedPhotonProducer") << " trackPairRef  " << trackPairRef.size() <<  "\n";
      reco::ConvertedPhoton  newCandidate(0, p4, vtx, convVtx);
      newCandidate.setP4(p4);
      outputConvPhotonCollection.push_back(newCandidate);
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
      outputConvPhotonCollection[iSC].setSuperCluster(scRef);
      
      iSC++;	
      myCands++;
       LogDebug("ConvertedPhotonProducer") << " Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	



    }

    
    
    ////
    
    
	lSC++;
    
  }


  //  Loop over SC in the Endcap and reconstruct converted photons


  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {
  
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
     LogDebug("ConvertedPhotonProducer") << " SC enery " << aClus->energy() <<  "  eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";
    
    
    ///// Find the +/- pairs
    //std::vector<std::vector<reco::Track> > allPairs = theTrackPairFinder_->run(outInTrkHandle,  inOutTrkHandle );
    std::vector<std::vector<reco::TransientTrack> > allPairs = theTrackPairFinder_->run(t_outInTrk,  t_inOutTrk );
    
     LogDebug("ConvertedPhotonProducer") << " Endcap  allPairs.size " << allPairs.size() << "\n";
    edm::RefVector<reco::TrackCollection> trackPairRef; 
    //std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;

    if ( allPairs.size() ) {
      
      for ( std::vector<std::vector<reco::TransientTrack> >::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
	 LogDebug("ConvertedPhotonProducer") << " Endcap  single pair size " << (*iPair).size() << "\n";  
	if (  (*iPair).size()  < 2) continue;
	
	CachingVertex theConversionVertex=theVertexFinder_->run(*iPair);
	if ( theConversionVertex.isValid() ) {
	   LogDebug("ConvertedPhotonProducer") << " conversion vertex position " << theConversionVertex.position() << "\n";
	} else {
	   LogDebug("ConvertedPhotonProducer") << " conversion vertex is not valid " << "\n";
	}
	
	//// Create a converted photon candidate per each track pair

	if ( theConversionVertex.isValid() ) {

	// final candidate 
	const reco::Particle::Point  vtx( 0, 0, 0 );
	//	const reco::Particle::Point  convVtx(   theConversionVertex->position().x(), theConversionVertex->position().y(),  theConversionVertex->position().z() );
	const reco::Particle::Point  convVtx(   theConversionVertex.position().x(), theConversionVertex.position().y(),  theConversionVertex.position().z() );
	 LogDebug("ConvertedPhotonProducer") << " SC energy " <<  aClus->energy() << "\n";
	math::XYZVector direction =aClus->position() - vtx;
	math::XYZVector momentum = direction.unit() * aClus->energy();
	const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );




    
	// Loop over tracksin the pair for deugging	
	trackPairRef.clear();
	for ( std::vector<reco::TransientTrack>::const_iterator iTk=(*iPair).begin(); iTk!=(*iPair).end(); ++iTk) {
	   LogDebug("ConvertedPhotonProducer") << " Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
	  
	  reco::TrackRef myTkRef= iTk->persistentTrackRef(); 
	   LogDebug("ConvertedPhotonProducer") << " Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	  
	  trackPairRef.push_back(myTkRef);
	}
	


	 LogDebug("ConvertedPhotonProducer") << " trackPairRef  " << trackPairRef.size() <<  "\n";

	reco::ConvertedPhoton  newCandidate(0, p4, vtx, convVtx);
        newCandidate.setP4(p4);
	outputConvPhotonCollection.push_back(newCandidate);
	// set the reference to the SC
	reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
	outputConvPhotonCollection[iSC].setSuperCluster(scRef);
	// set the reference to the tracks
        outputConvPhotonCollection[iSC].setTrackPairRef( trackPairRef);

	/*
	reco::SuperClusterRef scRef(scEndcapHandle, lSC);
	reco::ConvertedPhoton  newCandidate(scRef,  0, p4, vtx, convVtx);
	//reco::ConvertedPhoton  newCandidate(scRef, trackPairRef, 0, p4, vtx, convVtx);
	outputConvPhotonCollection.push_back(newCandidate);
	*/
	
	iSC++;      
	myCands++;
	 LogDebug("ConvertedPhotonProducer") << " Put the ConvertedPhotonCollection a candidate in the Endcap  " << "\n";
	
	}
	
      }
      
    } else {
       LogDebug("ConvertedPhotonProducer") << " GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      const reco::Particle::Point  vtx( 0, 0, 0 );
      const reco::Particle::Point  convVtx(  0, 0, 0 );
       LogDebug("ConvertedPhotonProducer") << " SC energy " <<  aClus->energy() << "\n";
      math::XYZVector direction =aClus->position() - vtx;
      math::XYZVector momentum = direction.unit() * aClus->energy();
      const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

       LogDebug("ConvertedPhotonProducer") << " trackPairRef  " << trackPairRef.size() <<  "\n";
      reco::ConvertedPhoton  newCandidate(0, p4, vtx, convVtx);
      newCandidate.setP4(p4);
      outputConvPhotonCollection.push_back(newCandidate);
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
      outputConvPhotonCollection[iSC].setSuperCluster(scRef);
      
      iSC++;	
      myCands++;
       LogDebug("ConvertedPhotonProducer") << " Put the ConvertedPhotonCollection a candidate in the Endcap " << "\n";
	


    }
    
   
	lSC++;
	
     
      
  }
  


  // put the product in the event
  
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
   LogDebug("ConvertedPhotonProducer") << " Putting in the event  " << myCands << "  converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";  
  theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);





}
