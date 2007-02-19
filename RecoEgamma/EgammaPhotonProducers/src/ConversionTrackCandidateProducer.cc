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
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateSuperClusterAssociation.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//  Abstract classes for the conversion tracking components
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackCandidateProducer.h"
//
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"

ConversionTrackCandidateProducer::ConversionTrackCandidateProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theNavigationSchool_(0), 
  theOutInSeedFinder_(0), 
  theOutInTrackFinder_(0), 
  theInOutSeedFinder_(0),
  theInOutTrackFinder_(0),
  isInitialized(0)

{


  LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer CTOR " << "\n";
  
   
  // use onfiguration file to setup input/output collection names
 
  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<std::string>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<std::string>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");
  
  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  
  OutInTrackCandidateCollection_ = conf_.getParameter<std::string>("outInTrackCandidateCollection");
  InOutTrackCandidateCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateCollection");

  OutInTrackSuperClusterAssociationCollection_ = conf_.getParameter<std::string>("outInTrackCandidateSCAssociationCollection");
  InOutTrackSuperClusterAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateSCAssociationCollection");

  // Register the product
  produces< TrackCandidateCollection > (OutInTrackCandidateCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateCollection_);
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( OutInTrackSuperClusterAssociationCollection_);
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( InOutTrackSuperClusterAssociationCollection_);


}

ConversionTrackCandidateProducer::~ConversionTrackCandidateProducer() {


  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  
  delete theInOutTrackFinder_;

}


void  ConversionTrackCandidateProducer::beginJob (edm::EventSetup const & theEventSetup) {

  //get magnetic field
  edm::LogInfo("ConversionTrackCandidateProducer") << " get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


  theEventSetup .get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker_ );


  // get the measurement tracker   
  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  theEventSetup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  theMeasurementTracker_ = measurementTrackerHandle.product();
  
  theLayerMeasurements_  = new LayerMeasurements(theMeasurementTracker_);
  theNavigationSchool_   = new SimpleNavigationSchool( &(*theGeomSearchTracker_)  , &(*theMF_));
  NavigationSetter setter( *theNavigationSchool_);
  
  // get the Out In Seed Finder  
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the OutInSeedFinder" << "\n";
  theOutInSeedFinder_ = new OutInConversionSeedFinder (   &(*theMF_) ,  theMeasurementTracker_ );
  
  // get the Out In Track Finder
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the OutInTrackFinder" << "\n";
  theOutInTrackFinder_ = new OutInConversionTrackFinder ( theEventSetup, conf_, &(*theMF_),  theMeasurementTracker_  );
  
  
  // get the In Out Seed Finder  
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the InOutSeedFinder" << "\n";
  theInOutSeedFinder_ = new InOutConversionSeedFinder (  &(*theMF_) ,  theMeasurementTracker_ );
  
  
  
  // get the In Out Track Finder
  edm::LogInfo("ConversionTrackCandidateProducer") << " get the InOutTrackFinder" << "\n";
  theInOutTrackFinder_ = new InOutConversionTrackFinder ( theEventSetup, conf_, &(*theMF_),  theMeasurementTracker_  );
  
  
}


void ConversionTrackCandidateProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {
  
  using namespace edm;
  
  edm::LogInfo("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << "\n";
  LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Analyzing event number " <<   theEvent.id() << "\n";
  
  
  // Update MeasurementTracker
  theMeasurementTracker_->update(theEvent);
  
  
  
  //
  // create empty output collections
  //
  //  Out In Track Candidates
  std::auto_ptr<TrackCandidateCollection> outInTrackCandidate_p(new TrackCandidateCollection); 
  //  In Out  Track Candidates
  std::auto_ptr<TrackCandidateCollection> inOutTrackCandidate_p(new TrackCandidateCollection); 
  //   Track Candidate  Super Cluster Association
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> outInAssoc_p(new reco::TrackCandidateSuperClusterAssociationCollection);
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> inOutAssoc_p(new reco::TrackCandidateSuperClusterAssociationCollection);


   
  // Get the basic cluster collection in the Barrel 
  edm::Handle<reco::BasicClusterCollection> bcBarrelHandle;
  theEvent.getByLabel(bcProducer_, bcBarrelCollection_, bcBarrelHandle);
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Trying to access basic cluster collection in the Barrel from my Producer " << "\n";
  reco::BasicClusterCollection clusterCollectionBarrel = *(bcBarrelHandle.product());
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer basic cluster collection size  " << clusterCollectionBarrel.size() << "\n";
  
  
  
  // Get the basic cluster collection in the Endcap 
  edm::Handle<reco::BasicClusterCollection> bcEndcapHandle;
  theEvent.getByLabel(bcProducer_, bcEndcapCollection_, bcEndcapHandle);
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Trying to access basic cluster collection in the Endcap from my Producer " << "\n";
  reco::BasicClusterCollection clusterCollectionEndcap = *(bcEndcapHandle.product());
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer basic cluster collection size  " << clusterCollectionEndcap.size() << "\n";
  
  
  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Trying to access " << scHybridBarrelCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer barrel  SC collection size  " << scBarrelCollection.size() << "\n";
  
  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Trying to access " <<scIslandEndcapCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
 LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Endcap SC collection size  " << scEndcapCollection.size() << "\n";
  

  /////////////


  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForOutIn;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForInOut;  
  
  
  //  Loop over SC in the barrel and reconstruct converted photons
  int iSC=0; // index in product collection
  int lSC=0; // local index on barrel

  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); ++aClus) {
  
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
    LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer  SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() <<  " Energy " <<  aClus->energy() << "\n";

    theOutInSeedFinder_->setCandidate(*aClus);
    theOutInSeedFinder_->makeSeeds(  clusterCollectionBarrel );


    //    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidate_p, *outInAssoc, iSC);    
    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidate_p);    

    theInOutSeedFinder_->setCandidate(*aClus);
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds(  clusterCollectionBarrel);
    

    // std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidate_p, *inOutAssoc, iSC);    
    std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidate_p); 


    // Debug
    LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer  theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size barrel " << (*outInTrackCandidate_p).size() << " in out track size " << (*inOutTrackCandidate_p).size() <<   "\n";


    //////////// Fill vectors of Ref to SC to be used for the Track-SC association
    reco::SuperClusterRef scRefOutIn(reco::SuperClusterRef(scBarrelHandle, lSC));
    reco::SuperClusterRef scRefInOut(reco::SuperClusterRef(scBarrelHandle, lSC));


    for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it !=  theOutInTracks.end(); ++it) {
      vecOfSCRefForOutIn.push_back( scRefOutIn );
           
      LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Barrel OutIn Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }

    for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it !=  theInOutTracks.end(); ++it) {
      vecOfSCRefForInOut.push_back( scRefInOut );

      LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Barrel InOut Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }




    iSC++;
    lSC++;
  }


  //  Loop over SC in the Endcap and reconstruct tracks from converted photons
  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); ++aClus) {
  
    //    if ( abs( aClus->eta() ) > 0.9 ) return; 
    LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() << " Energy " <<  aClus->energy() << "\n";

    theOutInSeedFinder_->setCandidate(*aClus);
    theOutInSeedFinder_->makeSeeds(  clusterCollectionEndcap );
  
 

    //    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidate_p, *outInAssoc, iSC);    
    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidate_p);    
    
    theInOutSeedFinder_->setCandidate(*aClus);
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds(  clusterCollectionEndcap );
    LogDebug("ConversionTrackCandidateProducer" << " ConversionTrackCandidateProducer inout seedfinding ended. Going to make the inout tracks " << "\n";

    // std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidate_p, *inOutAssoc, iSC);        
    std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidate_p); 


   LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size endcap " << (*outInTrackCandidate_p).size() << " in out track size " << (*inOutTrackCandidate_p).size() <<   "\n";
    

    //////////// Fill vectors of Ref to SC to be used for the Track-SC association
   
    reco::SuperClusterRef scRefInOut(reco::SuperClusterRef(scEndcapHandle, lSC));
    reco::SuperClusterRef scRefOutIn(reco::SuperClusterRef(scEndcapHandle, lSC));

    for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it !=  theOutInTracks.end(); ++it) {

      vecOfSCRefForOutIn.push_back( scRefOutIn );
      LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Endcap OutIn Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }

    for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it !=  theInOutTracks.end(); ++it) {
    
      vecOfSCRefForInOut.push_back( scRefInOut );
      LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Endcap InOut Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }


    //    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
    // vecOfSCRef.push_back( scRef );



    
    iSC++;
    lSC++; 
  }



  LogDebug("ConversionTrackCandidateProducer" << "  ConversionTrackCandidateProducer vecOfSCRefForOutIn size " << vecOfSCRefForOutIn.size() << " vecOfSCRefForInOut size " << vecOfSCRefForInOut.size()  << "\n"; 


  // put the product in the event

  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidate_p).size() << " Out In track Candidates " << "\n";
  edm::LogInfo("ConversionTrackCandidateProducer") << "Number of outInTrackCandidates: " <<  (*outInTrackCandidate_p).size() << "\n";
  const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackC = theEvent.put( outInTrackCandidate_p, OutInTrackCandidateCollection_ );
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer  refprodOutInTrackC size  " <<  (*(refprodOutInTrackC.product())).size()  <<  "\n";
  //
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidate_p).size() << " In Out track Candidates " <<  "\n";
  edm::LogInfo("ConversionTrackCandidateProducer") << "Number of inOutTrackCandidates: " <<  (*inOutTrackCandidate_p).size() << "\n";
  const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackC = theEvent.put( inOutTrackCandidate_p, InOutTrackCandidateCollection_ );
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer  refprodInOutTrackC size  " <<  (*(refprodInOutTrackC.product())).size()  <<  "\n";
  

  for (unsigned int i=0;i< vecOfSCRefForOutIn.size(); ++i) {
    outInAssoc_p->insert(edm::Ref<TrackCandidateCollection>(refprodOutInTrackC,i), vecOfSCRefForOutIn[i]  );
  }
  for (unsigned int i=0;i< vecOfSCRefForInOut.size(); ++i) {
    inOutAssoc_p->insert(edm::Ref<TrackCandidateCollection>(refprodInOutTrackC,i), vecOfSCRefForInOut[i]  );
  }
  
  
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC association: size  " <<  (*outInAssoc_p).size() << "\n";  
  theEvent.put( outInAssoc_p, OutInTrackSuperClusterAssociationCollection_);
  LogDebug("ConversionTrackCandidateProducer" << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC association: size  " <<  (*inOutAssoc_p).size() << "\n";  
  theEvent.put( inOutAssoc_p, InOutTrackSuperClusterAssociationCollection_);


  
}
