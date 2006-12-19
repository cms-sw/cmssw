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

  std::cout << " ConvertedPhotonProducer CTOR " << std::endl;

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

  //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << "get magnetic field" << "\n";
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


void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  
  edm::LogInfo("ConvertedPhotonProducer") << "Analyzing event number: " << theEvent.id() << "\n";
  std::cout << "ConvertedPhotonProducer:Analyzing event number " <<   theEvent.id() << std::endl;
  
  
  //
  // create empty output collections
  //

  // Converted photon candidates
  reco::ConvertedPhotonCollection outputConvPhotonCollection;
  std::auto_ptr< reco::ConvertedPhotonCollection > outputConvPhotonCollection_p(new reco::ConvertedPhotonCollection);
  std::cout << " Created empty ConvertedPhotonCollection size " <<   std::endl;


  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  std::cout << " Trying to access " << scHybridBarrelCollection_.c_str() << "  from my Producer " << std::endl;
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  std::cout << "barrel  SC collection size  " << scBarrelCollection.size() << std::endl;

  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  std::cout << " Trying to access " <<scIslandEndcapCollection_.c_str() << "  from my Producer " << std::endl;
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  std::cout << "Endcap SC collection size  " << scEndcapCollection.size() << std::endl;

  //// Get the Out In CKF tracks from conversions
  Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  std::cout << " ConvertedPhotonAnalyzer outInTrack collection size " << (*outInTrkHandle).size() << std::endl;

 // Loop over Out In Tracks
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkHandle).begin(); iTk !=  (*outInTrkHandle).end(); iTk++) {
    std::cout << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << std::endl;  

    std::cout << " Out In Track Extra inner momentum  " << iTk->extra()->outerMomentum() << std::endl;  
   
  }


  //// Get the In Out  CKF tracks from conversions
  Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  std::cout << " ConvertedPhotonAnalyzer inOutTrack collection size " << (*inOutTrkHandle).size() << std::endl;

  // Transform Track into TransientTrack (needed by the Vertex fitter)

  edm::ESHandle<TransientTrackBuilder> theB;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
  //do the conversion:
  std::vector<reco::TransientTrack> t_outInTrk = (*theB).build(outInTrkHandle );
  std::vector<reco::TransientTrack> t_inOutTrk = (*theB).build(inOutTrkHandle );



  ///// Find the +/- pairs
  //std::vector<std::vector<reco::Track> > allPairs = theTrackPairFinder_->run(outInTrkHandle,  inOutTrkHandle );
  std::vector<std::vector<reco::TransientTrack> > allPairs = theTrackPairFinder_->run(t_outInTrk,  t_inOutTrk );

  std::cout << "  ConvertedPhotonAnalyzer  allPairs.size " << allPairs.size() << std::endl;
  if ( allPairs.size() ) {
    
    for ( std::vector<std::vector<reco::TransientTrack> >::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
      std::cout << "  ConvertedPhotonAnalyzer  single pair size " << (*iPair).size() << std::endl;  
      
      theVertexFinder_->run(*iPair);

      for ( std::vector<reco::TransientTrack>::const_iterator iTk=(*iPair).begin(); iTk!=(*iPair).end(); ++iTk) {
	std::cout << "  ConvertedPhotonAnalyzer  Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << std::endl;  

        reco::TrackRef myTkRef= iTk->persistentTrackRef(); 
	std::cout << "  ConvertedPhotonAnalyzer  Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << std::endl;  


      }




    }




  } else {
    std::cout << "  ConvertedPhotonAnalyzer  GOLDEN PHOTON ?? Zero Tracks " <<  std::endl;  
  }



  reco::ConvertedPhotonCollection myConvPhotons;

  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  int iSC=0; // index in photon collection
  int lSC=0; // local index on barrel
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {
  
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
    std::cout << "  ConvertedPhotonProducer SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() << std::endl;

      const reco::Particle::Point  vtx( 0, 0, 0 );

      math::XYZVector direction =aClus->position() - vtx;
      math::XYZVector momentum = direction.unit() * aClus->energy();


      const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

      /// old implementation
      /*
      reco::ConvertedPhoton  newCandidate(0, p4, vtx);
      outputConvPhotonCollection.push_back(newCandidate);
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
      outputConvPhotonCollection[iSC].setSuperCluster(scRef);
      */


      reco::SuperClusterRef scRef(scBarrelHandle, lSC);
      //   reco::TrackRefVector  trkRef(  inOutTrkHandle, 
      reco::ConvertedPhoton  newCandidate(scRef, 0, p4, vtx);
      outputConvPhotonCollection.push_back(newCandidate);



      lSC++;
      iSC++;      
      myCands++;
      std::cout << " Put the ConvertedPhotonCollection a candidate in the Barrel " << std::endl;

      //    }





  }


  //  Loop over SC in the Endcap and reconstruct converted photons


  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {
  
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
    std::cout << "  ConvertedPhotonProducer SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() << std::endl;

      // final candidate 
      const reco::Particle::Point  vtx( 0, 0, 0 );
      math::XYZVector direction =aClus->position() - vtx;
      math::XYZVector momentum = direction.unit() * aClus->energy();

      const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

      // old implementation
      /*
      reco::ConvertedPhoton  newCandidate(0, p4, vtx);
      outputConvPhotonCollection.push_back(newCandidate);
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
      outputConvPhotonCollection[iSC].setSuperCluster(scRef);
      */

      reco::SuperClusterRef scRef(scEndcapHandle, lSC);
      // reco::TrackRefVector  trkRef(reco::TrackRefVector());
      reco::ConvertedPhoton  newCandidate(scRef,  0, p4, vtx);
      outputConvPhotonCollection.push_back(newCandidate);



      lSC++;
      iSC++;      
      myCands++;
      std::cout << " Put the ConvertedPhotonCollection a candidate in the Endcap  " << std::endl;

      //    }

  }



  // put the product in the event
  
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
  std::cout << "  ConvertedPhotonProducer Putting in the event  " << myCands << "  converted photon candidates " << (*outputConvPhotonCollection_p).size() << std::endl;  
  theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);





}
