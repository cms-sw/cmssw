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
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config) : 

  conf_(config), 
  theTrackPairFinder_(0), 
  theVertexFinder_(0), 
  theEcalImpactPositionFinder_(0)

{


  
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer CTOR " << "\n";
  
  
  
  // use onfiguration file to setup input collection names
  
  //bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<edm::InputTag>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<edm::InputTag>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<edm::InputTag>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<edm::InputTag>("scIslandEndcapProducer");
  
  //  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  // scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  

  conversionOITrackProducer_ = conf_.getParameter<std::string>("conversionOITrackProducer");
  conversionIOTrackProducer_ = conf_.getParameter<std::string>("conversionIOTrackProducer");
  
  outInTrackSCAssociationCollection_ = conf_.getParameter<std::string>("outInTrackSCAssociation");
  inOutTrackSCAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackSCAssociation");
  
  algoName_ = conf_.getParameter<std::string>( "AlgorithmName" );  

  recoverOneTrackCase_ = conf_.getParameter<bool>( "recoverOneTrackCase" );  
  dRForConversionRecovery_ = conf_.getParameter<double>("dRForConversionRecovery");
  deltaCotCut_ = conf_.getParameter<double>("deltaCotCut");
  minApproachDisCut_ = conf_.getParameter<double>("minApproachDisCut");
  
     
  // use onfiguration file to setup output collection names
  ConvertedPhotonCollection_     = conf_.getParameter<std::string>("convertedPhotonCollection");
  
  
  // Register the product
  produces< reco::ConversionCollection >(ConvertedPhotonCollection_);
  
  // instantiate the Track Pair Finder algorithm
  theTrackPairFinder_ = new ConversionTrackPairFinder ();
  // instantiate the Vertex Finder algorithm
  theVertexFinder_ = new ConversionVertexFinder ();


  // Inizilize my global event counter
  nEvt_=0;


  theEcalImpactPositionFinder_ =0;
  
}

ConvertedPhotonProducer::~ConvertedPhotonProducer() {}



void  ConvertedPhotonProducer::beginRun (edm::Run& r, edm::EventSetup const & theEventSetup) {
 

    //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  

  if ( ! theEcalImpactPositionFinder_) {
    
     // instantiate the algorithm for finding the position of the track extrapolation at the Ecal front face
    theEcalImpactPositionFinder_ = new   ConversionTrackEcalImpactPoint ( &(*theMF_) );
  }  
  theEcalImpactPositionFinder_->setMagneticField ( &(*theMF_) );


  // Transform Track into TransientTrack (needed by the Vertex fitter)
  theEventSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theTransientTrackBuilder_);



}


void  ConvertedPhotonProducer::endRun (edm::Run& r, edm::EventSetup const & theEventSetup) {
  delete theTrackPairFinder_;
  delete theVertexFinder_;
  delete theEcalImpactPositionFinder_; 
}


void  ConvertedPhotonProducer::endJob () {
  
  edm::LogInfo("ConvertedPhotonProducer") << " Analyzed " << nEvt_  << "\n";
  LogDebug("ConvertedPhotonProducer") << "ConvertedPhotonProducer::endJob Processed " << nEvt_ << " events " << "\n";
  
  
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
  bool validBarrelSCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scHybridBarrelProducer_.label();
    validBarrelSCHandle=false;
  }
   
  // Get the Super Cluster collection in the Endcap
  bool validEndcapSCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<scIslandEndcapProducer_.label();
    validEndcapSCHandle=false;
  }
  
    
  //// Get the Out In CKF tracks from conversions 
  bool validTrackInputs=true;
  Handle<reco::TrackCollection> outInTrkHandle;
  theEvent.getByLabel(conversionOITrackProducer_,  outInTrkHandle);
  if (!outInTrkHandle.isValid()) {
    std::cout << "Error! Can't get the conversionOITrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionOITrack " << "\n";
    validTrackInputs=false;
  }
  LogDebug("ConvertedPhotonProducer")<< "ConvertedPhotonProducer  outInTrack collection size " << (*outInTrkHandle).size() << "\n";
  
   
  //// Get the association map between CKF Out In tracks and the SC where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> outInTrkSCAssocHandle;
  theEvent.getByLabel( conversionOITrackProducer_, outInTrackSCAssociationCollection_, outInTrkSCAssocHandle);
  if (!outInTrkSCAssocHandle.isValid()) {
    std::cout << "Error! Can't get the product " <<  outInTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  outInTrackSCAssociationCollection_.c_str() <<"\n";
    validTrackInputs=false;
  }

  //// Get the In Out  CKF tracks from conversions 
  Handle<reco::TrackCollection> inOutTrkHandle;
  theEvent.getByLabel(conversionIOTrackProducer_, inOutTrkHandle);
  if (!inOutTrkHandle.isValid()) {
    std::cout << "Error! Can't get the conversionIOTrack " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the conversionIOTrack " << "\n";
    validTrackInputs=false;
  }
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer inOutTrack collection size " << (*inOutTrkHandle).size() << "\n";

  
  //// Get the association map between CKF in out tracks and the SC  where they originated
  Handle<reco::TrackCaloClusterPtrAssociation> inOutTrkSCAssocHandle;
  theEvent.getByLabel( conversionIOTrackProducer_, inOutTrackSCAssociationCollection_, inOutTrkSCAssocHandle);
  if (!inOutTrkSCAssocHandle.isValid()) {
    std::cout << "Error! Can't get the product " <<  inOutTrackSCAssociationCollection_.c_str() <<"\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product " <<  inOutTrackSCAssociationCollection_.c_str() <<"\n";
    validTrackInputs=false;
  }


  //// Get the generalTracks 
  Handle<reco::TrackCollection> generalTrkHandle;
  theEvent.getByLabel("generalTracks", generalTrkHandle);
  if (!generalTrkHandle.isValid()) {
    std::cout << "Error! Can't get the generalTracks " << "\n";
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the genralTracks " << "\n";
    validTrackInputs=false;
  }
  

  // Get the basic cluster collection in the Barrel 
  bool validBarrelBCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  theEvent.getByLabel( bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcBarrelCollection_.label();
     validBarrelBCHandle=false;
  }

    
  // Get the basic cluster collection in the Endcap 
  bool validEndcapBCHandle=true;
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;
  theEvent.getByLabel( bcEndcapCollection_, bcEndcapHandle);
  if (!bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") << "Error! Can't get the product "<<bcEndcapCollection_.label();
    validEndcapBCHandle=true;
  }
 
  


  if (  validTrackInputs ) {
    //do the conversion:
    std::vector<reco::TransientTrack> t_outInTrk = ( *theTransientTrackBuilder_ ).build(outInTrkHandle );
    std::vector<reco::TransientTrack> t_inOutTrk = ( *theTransientTrackBuilder_ ).build(inOutTrkHandle );
    
    
    ///// Find the +/- pairs
    //  std::map<std::vector<reco::TransientTrack>, const reco::SuperCluster*> allPairs;
    std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr> allPairs;
    allPairs = theTrackPairFinder_->run(t_outInTrk, outInTrkHandle, outInTrkSCAssocHandle, t_inOutTrk, inOutTrkHandle, inOutTrkSCAssocHandle  );
    LogDebug("ConvertedPhotonProducer")  << "ConvertedPhotonProducer  allPairs.size " << allPairs.size() << "\n";      

    buildCollections(scBarrelHandle, bcBarrelHandle,generalTrkHandle, allPairs, outputConvPhotonCollection);
    buildCollections(scEndcapHandle, bcEndcapHandle,generalTrkHandle, allPairs, outputConvPhotonCollection);
  }
  
  // put the product in the event
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(),outputConvPhotonCollection.end());
  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Putting in the event    converted photon candidates " << (*outputConvPhotonCollection_p).size() << "\n";  


  theEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);



  
}


void ConvertedPhotonProducer::buildCollections (  const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
						  const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
						  const edm::Handle<reco::TrackCollection>  & generalTrkHandle,
						  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>& allPairs,
                                                  reco::ConversionCollection & outputConvPhotonCollection)

{

  reco::Conversion::ConversionAlgorithm algo = reco::Conversion::algoByName(algoName_);
  std::vector<reco::TransientTrack> t_generalTrk = ( *theTransientTrackBuilder_ ).build(generalTrkHandle );

  //  Loop over SC in the barrel and reconstruct converted photons
  int myCands=0;
  reco::CaloClusterPtrVector scPtrVec;
  for (unsigned i = 0; i < scHandle->size(); ++i ) {

    reco::CaloClusterPtr aClus= scHandle->ptrAt(i);

    std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
    std::vector<math::XYZVector> trackPin;
    std::vector<math::XYZVector> trackPout;
    float minAppDist=-99;

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
	scPtrVec.clear();
       
	reco::CaloClusterPtr caloPtr=iPair->second;
	if ( !( aClus == caloPtr ) ) continue;
            
        scPtrVec.push_back(aClus);     
	nFound++;

	std::vector<math::XYZPoint> trkPositionAtEcal = theEcalImpactPositionFinder_->find(  iPair->first, bcHandle );
	std::vector<reco::CaloClusterPtr>  matchingBC = theEcalImpactPositionFinder_->matchingBC();
	

        minAppDist=-99;
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
	 
	  // Old TwoTrackMinimumDistance md;
	  // Old md.calculate  (  (iPair->first)[0].initialFreeState(),  (iPair->first)[1].initialFreeState() );
          // Old minAppDist = md.distance(); 
 
	
	

        
	

	/*
	for ( unsigned int i=0; i< matchingBC.size(); ++i) {
          if (  matchingBC[i].isNull() )  std::cout << " This ref to BC is null: skipping " <<  "\n";
          else 
	    std::cout << " BC energy " << matchingBC[i]->energy() <<  "\n";
	}
	*/


	//// loop over tracks in the pair  for creating a reference
	trackPairRef.clear();
        trackPin.clear();
	trackPout.clear();
      
	
	for ( std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin(); iTk!= (iPair->first).end(); ++iTk) {
	  LogDebug("ConvertedPhotonProducer")  << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
	  
	  const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
	  reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
	  
	  LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer Ref to Rec Tracks in the pair  charge " << myTkRef->charge() << " Num of RecHits " << myTkRef->recHitsSize() << " inner momentum " << myTkRef->innerMomentum() << "\n";  
	  if ( myTkRef->extra().isNonnull() ) {
	    trackPin.push_back(  myTkRef->innerMomentum());
	    trackPout.push_back(  myTkRef->outerMomentum());
	  }
	  trackPairRef.push_back(myTkRef);
	  
	}
	
	//	std::cout << " ConvertedPhotonProducer trackPin size " << trackPin.size() << std::endl;
	LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
        if( theConversionVertex.isValid() ) {
	  
	  LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " << theConversionVertex.position().x() << " " << theConversionVertex.position().y() << " " << theConversionVertex.position().z() << "\n";
	  
	}
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";



	//	std::cout << "  ConvertedPhotonProducer  algo name " << algoName_ << std::endl;

        
	minAppDist=calculateMinApproachDistance( trackPairRef[0],  trackPairRef[1]);

	reco::Conversion  newCandidate(scPtrVec,  trackPairRef,  trkPositionAtEcal, theConversionVertex, matchingBC,  minAppDist, trackPin, trackPout, algo );
	outputConvPhotonCollection.push_back(newCandidate);
	
	
	
	myCands++;
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
	
	} else {
	  
	  
	  //	  std::cout << "   ConvertedPhotonProducer case with only one track found " <<  "\n";
 
	    //std::cout << "   ConvertedPhotonProducer recovering one track " <<  "\n";
	    trackPairRef.clear();
	    trackPin.clear();
	    trackPout.clear();
	    std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin();
	    //std::cout  << "  ConvertedPhotonProducer Transient Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << " pt " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  	  
	    const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
	    reco::TrackRef myTk= ttt->persistentTrackRef(); 
	    if ( myTk->extra().isNonnull() ) {
	      trackPin.push_back(  myTk->innerMomentum());
	      trackPout.push_back(  myTk->outerMomentum());
	    }
	    trackPairRef.push_back(myTk);
	    //std::cout << " Provenance " << myTk->algoName() << std::endl;
	
	    if (  recoverOneTrackCase_ ) {    
	      float theta1 = myTk->innerMomentum().Theta();
	      float dCot=999.;
	      float dCotTheta=-999.;
	      reco::TrackRef goodRef;
	      std::vector<reco::TransientTrack>::const_iterator iGoodGenTran;
	      //	  for (unsigned int i=0; i< generalTrkHandle->size(); i++) {
	      // reco::TrackRef trRef(generalTrkHandle, i);
	      // if ( trRef->charge()*myTk->charge() > 0 ) continue;
	      // 
	      //  float theta2 = trRef->innerMomentum().Theta();
	      //  dCotTheta =  1./tan(theta1) - 1./tan(theta2) ;
	      //  if ( fabs(dCotTheta) < dCot ) {
	      //    dCot = fabs(dCotTheta);
	      //    goodRef = trRef;
	      //  }
	      //}
	      
	      for ( std::vector<reco::TransientTrack>::const_iterator iTran= t_generalTrk.begin(); iTran != t_generalTrk.end(); ++iTran) {
		const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTran->basicTransientTrack());
		reco::TrackRef trRef= ttt->persistentTrackRef(); 
		if ( trRef->charge()*myTk->charge() > 0 ) continue;
		float dEta =  trRef->eta() - myTk->eta();
		float dPhi =  trRef->phi() - myTk->phi();
		if ( sqrt (dEta*dEta + dPhi*dPhi) > dRForConversionRecovery_ ) continue; 
		float theta2 = trRef->innerMomentum().Theta();
		dCotTheta =  1./tan(theta1) - 1./tan(theta2) ;
		//    std::cout << "  ConvertedPhotonProducer general transient track charge " << trRef->charge() << " momentum " << trRef->innerMomentum() << " dcotTheta " << fabs(dCotTheta) << std::endl;
		if ( fabs(dCotTheta) < dCot ) {
		  dCot = fabs(dCotTheta);
		  goodRef = trRef;
		  iGoodGenTran=iTran;
		}
	      }
	      
	      if ( goodRef.isNonnull() ) {
		
		minAppDist=calculateMinApproachDistance( myTk, goodRef);
		
		// std::cout << "  ConvertedPhotonProducer chosen dCotTheta " <<  fabs(dCotTheta) << std::endl;
		if ( fabs(dCotTheta) < deltaCotCut_ && minAppDist > minApproachDisCut_ ) {
		  trackPin.push_back(  goodRef->innerMomentum());
		  trackPout.push_back(  goodRef->outerMomentum());
		  trackPairRef.push_back( goodRef );
		  //	    std::cout << " ConvertedPhotonProducer adding opposite charge track from generalTrackCollection charge " <<  goodRef ->charge() << " pt " << sqrt(goodRef->innerMomentum().perp2())  << " trackPairRef size " << trackPairRef.size() << std::endl;            
		  //std::cout << " Track Provenenance " << goodRef->algoName() << std::endl; 
		  std::vector<reco::TransientTrack> mypair;
		  mypair.push_back(*iTk); 
		  mypair.push_back(*iGoodGenTran); 
		  
		  try{
		    
		    TransientVertex trVtx=theVertexFinder_->run(mypair); 
		    theConversionVertex= trVtx;
		    
		  }
		  catch ( cms::Exception& e ) {
		    std::cout << " cms::Exception caught in ConvertedPhotonProducer::produce" << "\n" ;
		    edm::LogWarning(metname) << "cms::Exception caught in ConvertedPhotonProducer::produce\n"
					     << e.explainSelf();
		    
		  }
		} 
		
	      }	    
	      
	    } // bool On/Off one track case recovery using generalTracks  
	    reco::Conversion  newCandidate(scPtrVec,  trackPairRef,  trkPositionAtEcal, theConversionVertex, matchingBC,  minAppDist, trackPin, trackPout, algo );
	    outputConvPhotonCollection.push_back(newCandidate);
	      
	      
	      
	    
	} // case with only on track: looking in general tracks
	
	
	
	
      } 
      
    }
    




    /*
    if (  allPairs.size() ==0 || nFound ==0) {
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer GOLDEN PHOTON ?? Zero Tracks " <<  "\n";  
      LogDebug("ConvertedPhotonProducer")  << " ConvertedPhotonProducer SC energy " <<  aClus->energy() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer photon p4 " << p4  << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer vtx " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer trackPairRef  " << trackPairRef.size() <<  "\n";
      
      std::vector<math::XYZPoint> trkPositionAtEcal;
      std::vector<reco::CaloClusterPtr> matchingBC;

      scPtrVec.clear();
      scPtrVec.push_back(aClus);     
      reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
      outputConvPhotonCollection.push_back(newCandidate);

     
      if ( newCandidate.conversionVertex().isValid() ) 
	LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer theConversionVertex " <<  newCandidate.conversionVertex().position().x() << " " <<  newCandidate.conversionVertex().position().y() << " " <<  newCandidate.conversionVertex().position().z() << "\n";
      

     
      
      myCands++;
      LogDebug("ConvertedPhotonProducer") << " ConvertedPhotonProducer Put the ConvertedPhotonCollection a candidate in the Barrel " << "\n";
      
    }
    */
    
  

    
  }
  




}




float ConvertedPhotonProducer::calculateMinApproachDistance ( const reco::TrackRef& track1, const reco::TrackRef& track2) {
  float dist=9999.;

  double x1, x2, y1, y2;
  double xx_1 = track1->innerPosition().x(), yy_1 = track1->innerPosition().y(), zz_1 = track1->innerPosition().z();
  double xx_2 = track2->innerPosition().x(), yy_2 = track2->innerPosition().y(), zz_2 = track2->innerPosition().z();
  double radius1 = track1->innerMomentum().Rho()/(.3*(theMF_->inTesla(GlobalPoint(xx_1, yy_1, zz_1)).z()))*100;
  double radius2 = track2->innerMomentum().Rho()/(.3*(theMF_->inTesla(GlobalPoint(xx_2, yy_2, zz_2)).z()))*100;
  getCircleCenter(track1, radius1, x1, y1);
  getCircleCenter(track2, radius2, x2, y2);
  dist = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) - radius1 - radius2;
  
 return dist;

} 


void ConvertedPhotonProducer::getCircleCenter(const reco::TrackRef& tk, double r, double& x0, double& y0){
  double x1, y1, phi;
  x1 = tk->innerPosition().x();//inner position and inner momentum need track Extra!
  y1 = tk->innerPosition().y();
  phi = tk->innerMomentum().phi();
  const int charge = tk->charge();
  x0 = x1 + r*sin(phi)*charge;
  y0 = y1 - r*cos(phi)*charge;

}
