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
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

//
#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
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
  theEcalImpactPositionFinder_(0)

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
  

  conversionOITrackProducer_ = conf_.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_ = conf_.getParameter<std::string>("conversionIOTrackProducer");
  
  outInTrackSCAssociationCollection_ = conf_.getParameter<std::string>("outInTrackSCAssociation");
  inOutTrackSCAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackSCAssociation");
  
   
  // use onfiguration file to setup output collection names
  ConvertedPhotonCollection_     = conf_.getParameter<std::string>("convertedPhotonCollection");
  
  
  // Register the product
  produces< reco::ConversionCollection >(ConvertedPhotonCollection_);
  
  // instantiate the Track Pair Finder algorithm
  theTrackPairFinder_ = new ConversionTrackPairFinder ();
  // instantiate the Vertex Finder algorithm
  theVertexFinder_ = new ConversionVertexFinder ();
  
}

ConvertedPhotonProducer::~ConvertedPhotonProducer() {
  
  
  delete theTrackPairFinder_;
  delete theVertexFinder_;
  delete theEcalImpactPositionFinder_; 

}


void  ConvertedPhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {
  
  // Inizilize my global event counter
  nEvt_=0;
  
  //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  

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

  LogDebug("ConvertedPhotonProducer")   << "ConvertedPhotonProduce::produce event number " <<   theEvent.id() << " Global counter " << nEvt_ << "\n";
  
  //
  // create empty output collections
  //
  // Converted photon candidates
  reco::ConversionCollection outputConvPhotonCollection;
  std::auto_ptr<reco::ConversionCollection> outputConvPhotonCollection_p(new reco::ConversionCollection);

  
  // Get the Super Cluster collection in the Barrel
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scHybridBarrelCollection_.c_str();
    return;
  }
   
  // Get the Super Cluster collection in the Endcap
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scIslandEndcapCollection_.c_str();
    return;
  }
  
    
  //// Get the Out In CKF tracks from conversions 
  Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  if (!outInTrkHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrack " << "\n";
    return;
  }
  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer  outInTrack collection size " << (*outInTrkHandle).size() << "\n";
  
   
  //// Get the association map between CKF Out In tracks and the SC where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> outInTrkSCAssocHandle;
  theEvent.getByLabel( conversionOITrackProducer_, outInTrackSCAssociationCollection_, outInTrkSCAssocHandle);
  if (!outInTrkSCAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  outInTrackSCAssociationCollection_.c_str() <<"\n";
    return;
  }

  //// Get the In Out  CKF tracks from conversions 
  Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  if (!inOutTrkHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrack " << "\n";
    return;
  }
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";

  
  //// Get the association map between CKF in out tracks and the SC  where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> inOutTrkSCAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducer_, inOutTrackSCAssociationCollection_, inOutTrkSCAssocHandle);
  if (!inOutTrkSCAssocHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  inOutTrackSCAssociationCollection_.c_str() <<"\n";
    return;
  }
  

  // Get the basic cluster collection in the Barrel 
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByLabel(bcProducer_, bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcBarrelCollection_.c_str();
    return;
  }

    
  // Get the basic cluster collection in the Endcap 
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByLabel(bcProducer_, bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcEndcapCollection_.c_str();
    return;
  }
 
  
  // Transform Track into TransientTrack (needed by the Vertex fitter)
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder;
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder);
  //do the conversion:

  std::vector<reco::TransientTrack> t_outInTrk = ( *theTransientTrackBuilder ).build(outInTrkHandle );
  std::vector<reco::TransientTrack> t_inOutTrk = ( *theTransientTrackBuilder ).build(inOutTrkHandle );
  
  
  ///// Find the +/- pairs
  //  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*> allPairs;
  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr> allPairs;
  allPairs = theTrackPairFinder_->run(t_outInTrk, outInTrkHandle, outInTrkSCAssocHandle, t_inOutTrk, inOutTrkHandle, inOutTrkSCAssocHandle  );
  LogDebug("ConvertedPhotonProducer")  << "ConvertedPhotonProducer  allPairs.size " << allPairs.size() << "\n";      

  buildCollections(scBarrelHandle, bcBarrelHandle, allPairs, outputConvPhotonCollection);
  buildCollections(scEndcapHandle, bcEndcapHandle, allPairs, outputConvPhotonCollection);
  
  // put the product in the event
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Putting in the event    converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";  

  // edm::OrphanHandle<reco::ConversionCollection> cpHandle;
  theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);



  
}


void ConvertedPhotonProducer::buildCollections (  const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
						  const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
						  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>& allPairs,
                                                  reco::ConversionCollection & outputConvPhotonCollection)

{

  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  reco::CaloClusterPtrVector scPtrVec;
  for (unsigned i = 0; i < scHandle->size(); ++i ) {

    reco::CaloClusterPtr aClus= scHandle->ptrAt(i);

    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
    LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer SC energy " << aClus->energy() << " eta " <<  aClus->eta() << " phi " <<  aClus->phi() << "\n";

    
    //// Set here first quantities for the converted photon
    const reco::Particle::Point  vtx( 0, 0, 0 );
    reco::Vertex  theConversionVertex;
   
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );
    
    int nFound=0;    
    if ( allPairs.size() ) {

      nFound=0;
      for (  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>::const_iterator iPair= allPairs.begin(); iPair!= allPairs.end(); ++iPair ) {

       
	reco::CaloClusterPtr caloPtr=iPair->second;
	if ( !( aClus == caloPtr ) ) continue;
            
        scPtrVec.push_back(aClus);     
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
	

        
	std::vector<math::XYZPoint> trkPositionAtEcal = theEcalImpactPositionFinder_->find(  iPair->first, bcHandle );
	std::vector<reco::CaloClusterPtr>  matchingBC = theEcalImpactPositionFinder_->matchingBC();
	

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
		
	reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
	outputConvPhotonCollection.push_back(newCandidate);
	
	
	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	
      }
      
    }
     

    if (  allPairs.size() ==0 || nFound ==0) {
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      
      std::vector<math::XYZPoint> trkPositionAtEcal;
      std::vector<reco::CaloClusterPtr> matchingBC;


      reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
      outputConvPhotonCollection.push_back(newCandidate);

     
      if ( newCandidate.conversionVertex().isValid() ) 
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " <<  newCandidate.conversionVertex().position().x() << " " <<  newCandidate.conversionVertex().position().y() << " " <<  newCandidate.conversionVertex().position().z() << "\n";
      

     
      
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
      
    }

    
  

    
  }
  




}



