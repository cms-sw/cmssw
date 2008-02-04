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
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
//
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackEcalImpactPoint.h"
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
  theTrackPairFinder_(0), 
  theVertexFinder_(0), 
  theLayerMeasurements_(0), 
  theNavigationSchool_(0), 
  theEcalImpactPositionFinder_(0), 
  isInitialized(0)

{


  
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer CTOR " << "\n";
  
  
  
  // use onfiguration file to setup input collection names
  
  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<std::string>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<std::string>("bcEndcapCollection");


  photonProducer_         = conf_.getParameter<std::string>("photonProducer");
  photonCollection_       = conf_.getParameter<std::string>("photonCorrCollection");
  
  
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
  PhotonWithConversionsCollection_     = conf_.getParameter<std::string>("photonWithConversionsCollection");
  
  
  // Register the product
  produces< reco::ConversionCollection >(ConvertedPhotonCollection_);
  produces< reco::PhotonCollection >(PhotonWithConversionsCollection_);
  
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
  delete theEcalImpactPositionFinder_; 

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

  // instantiate the algorithm for finding the position of the track extrapolation at the Ecal front face
  theEcalImpactPositionFinder_ = new   ConversionTrackEcalImpactPoint ( &(*theMF_) );

  
  
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
  reco::ConversionCollection outputConvPhotonCollection;
  std::auto_ptr<reco::ConversionCollection> outputConvPhotonCollection_p(new reco::ConversionCollection);



  /// Photon completed with conversion info
  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr<reco::PhotonCollection> outputPhotonCollection_p(new reco::PhotonCollection);




  // Get the photons
  Handle<reco::PhotonCollection> photonHandle;
  theEvent.getByLabel(photonProducer_,photonCollection_,photonHandle);
  if (!photonHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<photonCollection_.c_str();
  }
  reco::PhotonCollection photonCollection = *(photonHandle.product());
  
  
  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scHybridBarrelCollection_.c_str();
  }
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer barrel  SC collection size  " << scBarrelCollection.size() << "\n";


  
  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scIslandEndcapCollection_.c_str();
  }
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap SC collection size  " << scEndcapCollection.size() << "\n";
  
    
  //// Get the Out In CKF tracks from conversions in the Barrel
  Handle<reco::TrackCollection> outInTrkBarrelHandle;
  theEvent.getByLabel(conversionOITrackProducerBarrel_,  outInTrkBarrelHandle);
  if (!outInTrkBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrackBarrel " << "\n";
  }
  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer Barrel outInTrack collection size " << (*outInTrkBarrelHandle).size() << "\n";
  //// Get the Out In CKF tracks from conversions in the Endcap
  Handle<reco::TrackCollection> outInTrkEndcapHandle;
  theEvent.getByLabel(conversionOITrackProducerEndcap_,  outInTrkEndcapHandle);
  if (!outInTrkEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrackEndcap " << "\n";
  }
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap outInTrack collection size " << (*outInTrkEndcapHandle).size() << "\n";
  
   
  //// Get the association map between CKF Out In tracks and the SC Barrel where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> outInTrkSCBarrelAssocHandle;
  theEvent.getByLabel( conversionOITrackProducerBarrel_, outInTrackSCBarrelAssociationCollection_, outInTrkSCBarrelAssocHandle);
  if (!outInTrkSCBarrelAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  outInTrackSCBarrelAssociationCollection_.c_str() <<"\n";
  }
  reco::TrackSuperClusterAssociationCollection outInTrackSCBarrelAss = *outInTrkSCBarrelAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer outInTrackBarrelSCAssoc collection size " << (*outInTrkSCBarrelAssocHandle).size() <<"\n";

  //// Get the association map between CKF Out In tracks and the SC Endcap  where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> outInTrkSCEndcapAssocHandle;
  theEvent.getByLabel( conversionOITrackProducerEndcap_, outInTrackSCEndcapAssociationCollection_, outInTrkSCEndcapAssocHandle);
  if (!outInTrkSCEndcapAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  outInTrackSCEndcapAssociationCollection_.c_str() <<"\n";
  }
  reco::TrackSuperClusterAssociationCollection outInTrackSCEndcapAss = *outInTrkSCEndcapAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer outInTrackEndcapSCAssoc collection size " << (*outInTrkSCEndcapAssocHandle).size() <<"\n";
  
  
  //// Get the In Out  CKF tracks from conversions in the Barrel
  Handle<reco::TrackCollection> inOutTrkBarrelHandle;
  theEvent.getByLabel(conversionIOTrackProducerBarrel_, inOutTrkBarrelHandle);
  if (!inOutTrkBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrackBarrel " << "\n";
  }
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Barre; inOutTrack collection size " << (*inOutTrkBarrelHandle).size() << "\n";

  //// Get the In Out  CKF tracks from conversions in the Endcap
  Handle<reco::TrackCollection> inOutTrkEndcapHandle;
  theEvent.getByLabel(conversionIOTrackProducerEndcap_, inOutTrkEndcapHandle);
  if (!inOutTrkEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrackEndcap " << "\n";
  }
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Endcap inOutTrack collection size " << (*inOutTrkEndcapHandle).size() << "\n";
  
  
  //// Get the association map between CKF in out tracks and the SC Barrel where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> inOutTrkSCBarrelAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducerBarrel_, inOutTrackSCBarrelAssociationCollection_, inOutTrkSCBarrelAssocHandle);
  if (!inOutTrkSCBarrelAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  inOutTrackSCBarrelAssociationCollection_.c_str() <<"\n";
  }
  reco::TrackSuperClusterAssociationCollection inOutTrackSCBarrelAss = *inOutTrkSCBarrelAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer inOutTrackSCBarrelAssoc collection size " << (*inOutTrkSCBarrelAssocHandle).size() <<"\n";
  //// Get the association map between CKF in out tracks and the SC Endcap where they originated
  Handle<reco::TrackSuperClusterAssociationCollection> inOutTrkSCEndcapAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducerEndcap_, inOutTrackSCEndcapAssociationCollection_, inOutTrkSCEndcapAssocHandle);
  if (!inOutTrkSCEndcapAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  inOutTrackSCEndcapAssociationCollection_.c_str() <<"\n";
  }
  reco::TrackSuperClusterAssociationCollection inOutTrackSCEndcapAss = *inOutTrkSCEndcapAssocHandle;  
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer inOutTrackSCBarrelAssoc collection size " << (*inOutTrkSCEndcapAssocHandle).size() <<"\n";
  
  

  // Get the basic cluster collection in the Barrel 
  edm::Handle<reco::BasicClusterCollection> bcBarrelHandle;
  theEvent.getByLabel(bcProducer_, bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcBarrelCollection_.c_str();
  }

    
  // Get the basic cluster collection in the Endcap 
  edm::Handle<reco::BasicClusterCollection> bcEndcapHandle;
  theEvent.getByLabel(bcProducer_, bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcEndcapCollection_.c_str();
  }
 
  

  
  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder);
  //do the conversion:
  std::vector<reco::TransientTrack> t_outInTrkBarrel = ( *theTransientTrackBuilder ).build(outInTrkBarrelHandle );
  std::vector<reco::TransientTrack> t_inOutTrkBarrel = ( *theTransientTrackBuilder ).build(inOutTrkBarrelHandle );
  std::vector<reco::TransientTrack> t_outInTrkEndcap = ( *theTransientTrackBuilder ).build(outInTrkEndcapHandle );
  std::vector<reco::TransientTrack> t_inOutTrkEndcap = ( *theTransientTrackBuilder ).build(inOutTrkEndcapHandle );
  
  
  reco::ConversionCollection myConvPhotons;
  
  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  
  int lSC=0; // local index on barrel
  std::vector<math::XYZPoint> trkPositionAtEcal; 
  // std::vector<reco::BasicCluster> matchingBC;
  std::vector<reco::BasicClusterRef> matchingBC;
 
  
  
  reco::SuperClusterCollection::iterator aClus;
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
  
  
  ///// Find the +/- pairs
  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*> allPairs;
  allPairs = theTrackPairFinder_->run(t_outInTrkBarrel, outInTrkBarrelHandle, outInTrkSCBarrelAssocHandle, t_inOutTrkBarrel, inOutTrkBarrelHandle, inOutTrkSCBarrelAssocHandle  );
  LogDebug("ConvertedPhotonProducer")  << "ConvertedPhotonProducer Barrel  allPairs.size " << allPairs.size() << "\n";      
  
  
  

  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); aClus++) {

    
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
    lSC++;

    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Barrel SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << " Pointer " << &(*scRef)  << "\n";

    
    //// Set here first quantities for the converted photon
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Vertex  theConversionVertex;
   
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
    
    int nFound=0;    
    if ( allPairs.size() ) {

      nFound=0;
      for (  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*>::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {


	
	LogDebug("ConvertedPhotonProducer")<< " ConvertedPhotonProducer Barrel single pair size " << (iPair->first).size() << " SC Energy " << (iPair->second)->energy() << " eta " << (iPair->second)->eta() << " phi " <<  (iPair->second)->phi() << " Pointer " << (iPair->second) << " ref " <<  &(*scRef) << "\n";  
       

        if (  iPair->second != &(*scRef) )  continue;
	//        std::cout <<    " ConvertedPhotonProducer Barrel  SC passing " <<    iPair->second  << " " << &(*scRef) << "\n";    
            
     
	nFound++;
	
	
	const string metname = "ConvertedPhotons|ConvertedPhotonProducer";
	if ( (iPair->first).size()  > 1 ) {
	  try{
	    
	    TransientVertex trVtx=theVertexFinder_->run(iPair->first); 
	    theConversionVertex= trVtx;
	    
	  }
	  catch ( cms::Exception& e ) {
	    std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
	    edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
				     << e.explainSelf();
	    
	  }
	  
	}
	

        
	std::vector<math::XYZPoint> trkPositionAtEcal = theEcalImpactPositionFinder_->find(  iPair->first, bcBarrelHandle );
	matchingBC = theEcalImpactPositionFinder_->matchingBC();
	/*
	for ( unsigned int i=0; i< matchingBC.size(); ++i) {
          if (  matchingBC[i].isNull() )  std::cout << " This ref to BC is null: skipping " <<  "\n";
          else 
	    std::cout << " BC energy " << matchingBC[i]->energy() <<  "\n";
	}
	*/


	//// loop over tracks in the pair  for creating a reference
	trackPairRef.clear();

	
	for ( std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin(); iTk!= (iPair->first).end(); ++iTk) {
	  LogDebug("ConvertedPhotonProducer")  << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
	  
	  const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
	  reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
	  
	  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	  
	  
	  trackPairRef.push_back(myTkRef);
	  
	}
	
	
	
	LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
        if( theConversionVertex.isValid() ) {
	  
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " << theConversionVertex.position().x() << " " << theConversionVertex.position().y() << " " << theConversionVertex.position().z() << "\n";
	  
	}
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
		
	reco::Conversion  newCandidate(scRef,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
	outputConvPhotonCollection.push_back(newCandidate);
	
	
	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	
      }
      
    }
      //} else {


    if (  allPairs.size() ==0 || nFound ==0) {
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      


      reco::Conversion  newCandidate(scRef,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
      outputConvPhotonCollection.push_back(newCandidate);

     
      if ( newCandidate.conversionVertex().isValid() ) 
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " <<  newCandidate.conversionVertex().position().x() << " " <<  newCandidate.conversionVertex().position().y() << " " <<  newCandidate.conversionVertex().position().z() << "\n";
      

     
      
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
      
    }

    
  

    
  }
  
  

  //  Loop over SC in the Endcap and reconstruct converted photons

  ///// Find the +/- pairs 
  allPairs = theTrackPairFinder_->run(t_outInTrkEndcap, outInTrkEndcapHandle, outInTrkSCEndcapAssocHandle, t_inOutTrkEndcap, inOutTrkEndcapHandle, inOutTrkSCEndcapAssocHandle  );
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap  allPairs.size " << allPairs.size() << "\n";


  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); aClus++) {

    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
    lSC++;    

    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;

    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer Endcap SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() <<  " Pointer " << &(*aClus) << "\n";    
    
    //// Set here first quantities for the converted photon
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Vertex  theConversionVertex;      
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
    

    int nFound=0;        
    if ( allPairs.size() ) {

      nFound=0;

      for (  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*>::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Endcap single pair size " << (iPair->first).size() << " SC Energy " << (iPair->second)->energy() << " eta " << (iPair->second)->eta() << " phi " <<  (iPair->second)->phi() << " Pointer " << (iPair->second) << " ref " <<  &(*scRef) << "\n";  
	

	if (  iPair->second != &(*scRef) )  continue;
	

	nFound++;
	
	
	const string metname = "ConvertedPhotons|ConvertedPhotonProducer";
	if ( (iPair->first).size()  > 1 ) {
	  try{
	    TransientVertex trVtx=theVertexFinder_->run(iPair->first); 
	    theConversionVertex= trVtx;
	    
	  }
	  catch ( cms::Exception& e ) {
	    std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
	    edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
				     << e.explainSelf();
	    
	  }
	  
	}
	
	
	
	
	
	std::vector<math::XYZPoint> trkPositionAtEcal = theEcalImpactPositionFinder_->find(  iPair->first, bcEndcapHandle );
	matchingBC = theEcalImpactPositionFinder_->matchingBC();

	/*  this was just for debug
	std::cout  << " ConvertedPhotonProducer Endcap trkPositionAtEcal size " << trkPositionAtEcal.size() << " ref to matchingBC size " <<  matchingBC.size() << "\n";
	for ( unsigned int i=0; i< matchingBC.size(); ++i) {
          if (  matchingBC[i].isNull() )  std::cout << " This ref to BC is null: skipping " <<  "\n";
          else 
	    std::cout << " BC energy " << matchingBC[i]->energy() <<  "\n";
	}
	*/


	//// loop over tracks in the pair for creating a reference
	trackPairRef.clear();
	
	
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
	if( theConversionVertex.isValid() ) {
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " << theConversionVertex.position().x() << " " << theConversionVertex.position().y() << " " << theConversionVertex.position().z() << "\n";
	}
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
	
	
	
	reco::Conversion  newCandidate(scRef,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC );
	outputConvPhotonCollection.push_back(newCandidate);
	
        if ( newCandidate.conversionVertex().isValid() ) 
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " <<  newCandidate.conversionVertex().position().x() << " " <<  newCandidate.conversionVertex().position().y() << " " <<  newCandidate.conversionVertex().position().z() << "\n";
	
	
	
	
	
	
	
	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Endcap " << "\n";
	
      }
      
    }      
    //    } else {
    
    
    if (  allPairs.size() ==0 || nFound ==0) {
      
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      
      
      
      reco::Conversion  newCandidate(scRef,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC );
      outputConvPhotonCollection.push_back(newCandidate);
            
      if ( newCandidate.conversionVertex().isValid() ) 
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " <<  newCandidate.conversionVertex().position().x() << " " <<  newCandidate.conversionVertex().position().y() << " " <<  newCandidate.conversionVertex().position().z() << "\n";
      
      
      
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Endcap " << "\n";
      
    }
    
    
    
    
  }
  
  
  
  
  // put the product in the event

  
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Putting in the event  " << myCands << "  converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";  
  edm::OrphanHandle<reco::ConversionCollection> cpHandle;
  cpHandle=theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);


  reco::ConversionCollection cpCollection= *(cpHandle.product());

  //  std::cout << " ConvertedPhotonProducer orhpan handle size " << cpCollection.size() << std::endl;
  //std::cout << " ConvertedPhotonProducer corrected photon size " << photonCorrCollection.size() << std::endl;   
  for(reco::PhotonCollection::iterator phoItr = photonCollection.begin(); phoItr != photonCollection.end(); phoItr++) {


    reco::Photon* photon= phoItr->clone();
    const reco::SuperCluster* aClus=&(*(photon->superCluster()));
    //    std::cout << "ConvertedPhotonProducer  Loop on  Photons: SC energy " <<  aClus->energy() << std::endl;
    
    
    int icp=0;    
    for( reco::ConversionCollection::iterator  itCP = cpCollection.begin(); itCP != cpCollection.end(); itCP++) {
      
      reco::ConversionRef cpRef(reco::ConversionRef(cpHandle,icp));
      icp++;      
      if ( &(*itCP->superCluster())  != &(*aClus)  ) continue; 
      if ( !(*itCP).isConverted() ) continue;  


      photon->addConversion(cpRef);     
 
      
    }		     
    
    outputPhotonCollection.push_back(*photon);    

  }


  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer Putting in the event  " <<  (*outputPhotonCollection_p).size() << " photons completed with conversions " << "\n";  
  theEvent.put( outputPhotonCollection_p, PhotonWithConversionsCollection_ );

  /*
  edm::OrphanHandle<reco::PhotonCollection> phoHandle;
  phoHandle=theEvent.put( outputPhotonCollection_p, PhotonWithConversionsCollection_ );
 
  reco::PhotonCollection newphoCollection= *(phoHandle.product());
  std::cout << " Newly created photon collection size " << newphoCollection.size() << std::endl;  
  for(reco::PhotonCollection::iterator phoItr = newphoCollection.begin(); phoItr != newphoCollection.end(); phoItr++) {

    std::cout << " Loop and  check ref to conversion size " << (*phoItr).conversions().size() << " r9 " << (*phoItr).r9() << " r19 " << (*phoItr).r19() << " 5x5 " << (*phoItr).e5x5() <<  std::endl;
    for ( int iCP=0; iCP<(*phoItr).conversions().size(); ++iCP) {
      std::cout << " Inv Mass " << (*phoItr).conversions()[iCP]->pairInvariantMass() << " dCotTheta " << (*phoItr).conversions()[iCP]->pairCotThetaSeparation() << " EoverP " <<  (*phoItr).conversions()[iCP]->EoverP()  <<  " primary vertex " << (*phoItr).conversions()[iCP]->zOfPrimaryVertexFromTracks() << " pairMomentum " << (*phoItr).conversions()[iCP]->pairMomentum()  <<  std::endl;
    }
  }
  */


  
}
