#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
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
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theNavigationSchool_(0), 
  isInitialized(0)

{


  
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer CTOR " << "\n";
  
  
  
  // use onfiguration file to setup input collection names
  
  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<std::string>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<std::string>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");
  
  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  
  conversionOITrackProducerBarrel_ = conf_.getParameter<std::string>("conversionOITrackProducerBarrel");
  conversionIOTrackProducerBarrel_ = conf_.getParameter<std::string>("conversionIOTrackProducerBarrel");
  
  conversionOITrackProducerEndcap_ = conf_.getParameter<std::string>("conversionOITrackProducerEndcap");
  conversionIOTrackProducerEndcap_ = conf_.getParameter<std::string>("conversionIOTrackProducerEndcap");
  
  outInTrackSCBarrelAssociationCollection_ = conf_.getParameter<std::string>("outInTrackSCBarrelAssociation");
  inOutTrackSCBarrelAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackSCBarrelAssociation");
  
  outInTrackSCEndcapAssociationCollection_ = conf_.getParameter<std::string>("outInTrackSCEndcapAssociation");
  inOutTrackSCEndcapAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackSCEndcapAssociation");
  
  
  barrelClusterShapeMapProducer_   = conf_.getParameter<std::string>("barrelClusterShapeMapProducer");
  barrelClusterShapeMapCollection_ = conf_.getParameter<std::string>("barrelClusterShapeMapCollection");
  endcapClusterShapeMapProducer_   = conf_.getParameter<std::string>("endcapClusterShapeMapProducer");
  endcapClusterShapeMapCollection_ = conf_.getParameter<std::string>("endcapClusterShapeMapCollection");
  
  
  
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
  delete theLayerMeasurements_;
  delete theNavigationSchool_;
  
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
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer::endJob Analyzed " << nEvt_ << " events " << "\n";
  
  
}

void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  nEvt_++;  
  LogInfo("ConvertedPhotonProducer") << "Analyzing event number: " << theEvent.id() << " Global counter " << nEvt_  << "\n";
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProduce::produce event number " <<   theEvent.id() << " Global counter " << nEvt_ << "\n";
  
  
  //
  // create empty output collections
  //
  
  // Converted photon candidates
  reco::ConvertedPhotonCollection outputConvPhotonCollection;
  std::auto_ptr< reco::ConvertedPhotonCollection > outputConvPhotonCollection_p(new reco::ConvertedPhotonCollection);
  LogDebug("ConvertedPhotonProducer")<< " ConvertedPhotonProducer Created empty ConvertedPhotonCollection size " <<   "\n";
  
  
  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Trying to access " << scHybridBarrelCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer barrel  SC collection size  " << scBarrelCollection.size() << "\n";
  
  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Trying to access " <<scIslandEndcapCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap SC collection size  " << scEndcapCollection.size() << "\n";
  
  
  // Get ClusterShape association maps
  Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle;
  theEvent.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_, barrelClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle;
  //
  Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle;
  theEvent.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle;
  
  
  
  
  //// Get the Out In CKF tracks from conversions in the Barrel
  Handle<reco::TrackCollection> outInTrkBarrelHandle;
  theEvent.getByLabel(conversionOITrackProducerBarrel_,  outInTrkBarrelHandle);
  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer Barrel outInTrack collection size " << (*outInTrkBarrelHandle).size() << "\n";
  //// Get the Out In CKF tracks from conversions in the Endcap
  Handle<reco::TrackCollection> outInTrkEndcapHandle;
  theEvent.getByLabel(conversionOITrackProducerEndcap_,  outInTrkEndcapHandle);
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap outInTrack collection size " << (*outInTrkEndcapHandle).size() << "\n";
  
  // Loop over Out In Tracks in the Barrel
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkBarrelHandle).begin(); iTk !=  (*outInTrkBarrelHandle).end(); iTk++) { 
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel Out In Track Extra inner momentum  " << iTk->extra()->innerMomentum() << "\n";  
    
  }
  // Loop over Out In Tracks in the Endcap
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrkEndcapHandle).begin(); iTk !=  (*outInTrkEndcapHandle).end(); iTk++) {
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap Out In Track Extra inner momentum  " << iTk->extra()->innerMomentum() << "\n";  
    
  }
  
  //// Get the association map between CKF Out In tracks and the SC Barrel where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> outInTrkSCBarrelAssocHandle;
  theEvent.getByLabel( conversionOITrackProducerBarrel_, outInTrackSCBarrelAssociationCollection_, outInTrkSCBarrelAssocHandle);
  reco::TrackSuperClusterAssociationCollection outInTrackSCBarrelAss = *outInTrkSCBarrelAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer outInTrackBarrelSCAssoc collection size " << (*outInTrkSCBarrelAssocHandle).size() <<"\n";
  //// Get the association map between CKF Out In tracks and the SC Endcap  where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> outInTrkSCEndcapAssocHandle;
  theEvent.getByLabel( conversionOITrackProducerEndcap_, outInTrackSCEndcapAssociationCollection_, outInTrkSCEndcapAssocHandle);
  reco::TrackSuperClusterAssociationCollection outInTrackSCEndcapAss = *outInTrkSCEndcapAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer outInTrackEndcapSCAssoc collection size " << (*outInTrkSCEndcapAssocHandle).size() <<"\n";
  
  
  //// Get the In Out  CKF tracks from conversions in the Barrel
  Handle<reco::TrackCollection> inOutTrkBarrelHandle;
  theEvent.getByLabel(conversionIOTrackProducerBarrel_, inOutTrkBarrelHandle);
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Barre; inOutTrack collection size " << (*inOutTrkBarrelHandle).size() << "\n";
  //// Get the In Out  CKF tracks from conversions in the Endcap
  Handle<reco::TrackCollection> inOutTrkEndcapHandle;
  theEvent.getByLabel(conversionIOTrackProducerEndcap_, inOutTrkEndcapHandle);
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Endcap inOutTrack collection size " << (*inOutTrkEndcapHandle).size() << "\n";
  // Loop over In Out  Tracks in the Barrel
  for( reco::TrackCollection::const_iterator  iTk =  (*inOutTrkBarrelHandle).begin(); iTk !=  (*inOutTrkBarrelHandle).end(); iTk++) {
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel In Out  Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel In Out  Track Extra inner momentum  " << iTk->extra()->innerMomentum() << "\n";  
    
  }
  // Loop over In Out  Tracks in the Endcap
  for( reco::TrackCollection::const_iterator  iTk =  (*inOutTrkEndcapHandle).begin(); iTk !=  (*inOutTrkEndcapHandle).end(); iTk++) {
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap In Out  Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap In Out  Track Extra inner momentum  " << iTk->extra()->innerMomentum() << "\n";  
    
  }
  
  
  //// Get the association map between CKF in out tracks and the SC Barrel where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> inOutTrkSCBarrelAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducerBarrel_, inOutTrackSCBarrelAssociationCollection_, inOutTrkSCBarrelAssocHandle);
  reco::TrackSuperClusterAssociationCollection inOutTrackSCBarrelAss = *inOutTrkSCBarrelAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer inOutTrackSCBarrelAssoc collection size " << (*inOutTrkSCBarrelAssocHandle).size() <<"\n";
  //// Get the association map between CKF in out tracks and the SC Endcap where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> inOutTrkSCEndcapAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducerEndcap_, inOutTrackSCEndcapAssociationCollection_, inOutTrkSCEndcapAssocHandle);
  reco::TrackSuperClusterAssociationCollection inOutTrackSCEndcapAss = *inOutTrkSCEndcapAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer inOutTrackSCBarrelAssoc collection size " << (*inOutTrkSCEndcapAssocHandle).size() <<"\n";
  
  
  
  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder);
  //do the conversion:
  std::vector<reco::TransientTrack> t_outInTrkBarrel = ( *theTransientTrackBuilder ).build(outInTrkBarrelHandle );
  std::vector<reco::TransientTrack> t_inOutTrkBarrel = ( *theTransientTrackBuilder ).build(inOutTrkBarrelHandle );
  std::vector<reco::TransientTrack> t_outInTrkEndcap = ( *theTransientTrackBuilder ).build(outInTrkEndcapHandle );
  std::vector<reco::TransientTrack> t_inOutTrkEndcap = ( *theTransientTrackBuilder ).build(inOutTrkEndcapHandle );
  
  
  
  reco::ConvertedPhotonCollection myConvPhotons;
  
  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  int iSC=0; // index in photon collection
  int lSC=0; // local index on barrel
  std::vector<math::XYZPoint> trkPositionAtEcal;
  
  
  
  reco::SuperClusterCollection::iterator aClus;
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
  
  
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";
    seedShpItr = barrelClShpMap.find(aClus->seed());
    assert(seedShpItr != barrelClShpMap.end());
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
    double r9 = seedShapeRef->e3x3()/(aClus->rawEnergy()+aClus->preshowerEnergy());
    
    
    ///// Find the +/- pairs
    std::map<std::vector<reco::TransientTrack>, reco::SuperCluster> allPairs = theTrackPairFinder_->run(t_outInTrkBarrel, outInTrkBarrelHandle, outInTrkSCBarrelAssocHandle, t_inOutTrkBarrel, inOutTrkBarrelHandle, inOutTrkSCBarrelAssocHandle  );
    
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel  allPairs.size " << allPairs.size() << "\n";
    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
    
    //// Set here first quantities for the converted photon
    const reco::Particle::Point  vtx( 0, 0, 0 );
    math::XYZPoint convVtx(0.,0.,0.);
    
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
    
    
    if ( allPairs.size() ) {
      
      //      for ( std::vector<std::vector<reco::TransientTrack> >::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
      for (  std::map<std::vector<reco::TransientTrack>, reco::SuperCluster>::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Barrel single pair size " << (iPair->first).size() << " SC Energy " << (iPair->second).energy() << " eta " << (iPair->second).eta() << " phi " <<  (iPair->second).phi() << "\n";  
	
	
	
	
	if( !( (  fabs(  (iPair->second).energy()  - aClus->energy()  ) < 0.001 ) &&  
	       (  fabs(  (iPair->second).eta()     - aClus->eta()     ) < 0.001 )      &&
	       (  fabs(  (iPair->second).phi()     -  aClus->phi()    ) < 0.001  ) ) )   continue;
	
	
	
	CachingVertex theConversionVertex;
	
	const string metname = "ConvertedPhotons|ConvertedPhotonProducer";
	if ( (iPair->first).size()  > 1 ) {
	  try{
	    
	    theConversionVertex=theVertexFinder_->run(iPair->first);
	  }
	  catch ( cms::Exception& e ) {
	    LogDebug("ConvertedPhotonProducer") << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
	    edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
				     << e.explainSelf();
	    
	  }
	  
	}
	
	
	
	
        if ( theConversionVertex.isValid() ) {	
	  convVtx.SetXYZ( theConversionVertex.position().x(), theConversionVertex.position().y(),  theConversionVertex.position().z() );
	  LogDebug("ConvertedPhotonProducer") << "  ConvertedPhotonProducer conversion vertex position " << theConversionVertex.position() << "\n";
	} else {
	  LogDebug("ConvertedPhotonProducer") << "  ConvertedPhotonProducer conversion vertex is not valid set the position to (0,0,0) " << "\n";
	}
	
	
	//// loop over tracks in the pair  for creating a reference
	trackPairRef.clear();
	trkPositionAtEcal.clear();
	
	for ( std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin(); iTk!= (iPair->first).end(); ++iTk) {
	  LogDebug("ConvertedPhotonProducer") << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
	  
	  const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
	  reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
	  
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	  
	  
	  trackPairRef.push_back(myTkRef);
	  
	}
	
	
	
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer convVtx " << convVtx.x() << " " << convVtx.y() << " " << convVtx.z() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
	
	
	
	reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
	reco::ConvertedPhoton  newCandidate(scRef,  trackPairRef, 0, p4, r9,  trkPositionAtEcal, vtx, convVtx);
	outputConvPhotonCollection.push_back(newCandidate);
	
	
	iSC++;	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	
      }
      
    } else {
      
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer convVtx " << convVtx.x() << " " << convVtx.y() << " " << convVtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      
      
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
      reco::ConvertedPhoton  newCandidate(scRef,  trackPairRef, 0, p4, r9,  trkPositionAtEcal, vtx, convVtx);
      outputConvPhotonCollection.push_back(newCandidate);
      
      iSC++;	
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
      
    }
    
    
    lSC++;
    
  }
  
  
  //  Loop over SC in the Endcap and reconstruct converted photons
  
  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {
    seedShpItr = endcapClShpMap.find(aClus->seed());
    assert(seedShpItr != endcapClShpMap.end());
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
    double r9 = seedShapeRef->e3x3()/(aClus->rawEnergy()+aClus->preshowerEnergy());
    
    
    
    LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";
    
    
    ///// Find the +/- pairs
    std::map<std::vector<reco::TransientTrack>, reco::SuperCluster> allPairs = theTrackPairFinder_->run(t_outInTrkEndcap, outInTrkEndcapHandle, outInTrkSCEndcapAssocHandle, t_inOutTrkEndcap, inOutTrkEndcapHandle, inOutTrkSCEndcapAssocHandle  );
    
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap  allPairs.size " << allPairs.size() << "\n";
    
    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
    
    //// Set here first quantities for the converted photon
    const reco::Particle::Point  vtx( 0, 0, 0 );
    math::XYZPoint convVtx(0.,0.,0.);
    
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
    
    
    if ( allPairs.size() ) {
      for (  std::map<std::vector<reco::TransientTrack>, reco::SuperCluster>::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Endcap single pair size " << (iPair->first).size() << " SC Energy " << (iPair->second).energy() << " eta " << (iPair->second).eta() << " phi " <<  (iPair->second).phi() << "\n";  
	
	
	
	if( !( (  fabs(  (iPair->second).energy()  - aClus->energy()  ) < 0.001 ) &&  
	       (  fabs(  (iPair->second).eta()     - aClus->eta()     ) < 0.001 )      &&
	       (  fabs(  (iPair->second).phi()     -  aClus->phi()    ) < 0.001  ) ) )   continue;
	
	
	CachingVertex theConversionVertex;
	const string metname = "ConvertedPhotons|ConvertedPhotonProducer";
	if ( (iPair->first).size()  > 1 ) {
	  try{
	    
	    theConversionVertex=theVertexFinder_->run(iPair->first);
	  }
	  catch ( cms::Exception& e ) {
	    LogDebug("ConvertedPhotonProducer") << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
	    edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
				     << e.explainSelf();
	    
	  }
	  
	}

	
	
        if ( theConversionVertex.isValid() ) {	
	  convVtx.SetXYZ( theConversionVertex.position().x(), theConversionVertex.position().y(),  theConversionVertex.position().z() );
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer conversion vertex position " << theConversionVertex.position() << "\n";
	} else {
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer conversion vertex is not valid " << "\n";
	}
	
	
	//// loop over tracks in the pair for creating a reference
	trackPairRef.clear();
	trkPositionAtEcal.clear();
	
	for ( std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin(); iTk!=(iPair->first).end(); ++iTk) {
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
	  
	  
	  const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
	  reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
	  
	  
	  
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	  
	  
	  trackPairRef.push_back(myTkRef);
	  
	}
	
	
	
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer convVtx " << convVtx.x() << " " << convVtx.y() << " " << convVtx.z() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
	
	
	reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
	reco::ConvertedPhoton  newCandidate(scRef,  trackPairRef, 0, p4, r9,  trkPositionAtEcal,  vtx, convVtx);
	outputConvPhotonCollection.push_back(newCandidate);
	
	
	iSC++;	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Endcap " << "\n";
	
      }
      
    } else {
      
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer convVtx " << convVtx.x() << " " << convVtx.y() << " " << convVtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      
      
      reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
      reco::ConvertedPhoton  newCandidate(scRef,  trackPairRef, 0, p4, r9, trkPositionAtEcal, vtx, convVtx);
      outputConvPhotonCollection.push_back(newCandidate);
      
      iSC++;	
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Endcap " << "\n";
      
    }
    
    
    lSC++;
    
  }
  
  
  
  
  // put the product in the event
  
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Putting in the event  " << myCands << "  converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";  
  theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);
  
  
  
  
  
}
