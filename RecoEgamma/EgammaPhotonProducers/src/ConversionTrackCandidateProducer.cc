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


  //  LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer CTOR " << "\n";
 LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer CTOR " << "\n";
  
   
  // use onfiguration file to setup input/output collection names
 
  bcProducer_             = conf_.getParameter<std::string>("bcProducer");
  bcBarrelCollection_     = conf_.getParameter<std::string>("bcBarrelCollection");
  bcEndcapCollection_     = conf_.getParameter<std::string>("bcEndcapCollection");
  
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");
  
  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  
  OutInTrackCandidateBarrelCollection_ = conf_.getParameter<std::string>("outInTrackCandidateBarrelCollection");
  InOutTrackCandidateBarrelCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateBarrelCollection");

  OutInTrackCandidateEndcapCollection_ = conf_.getParameter<std::string>("outInTrackCandidateEndcapCollection");
  InOutTrackCandidateEndcapCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateEndcapCollection");

  OutInTrackSuperClusterBarrelAssociationCollection_ = conf_.getParameter<std::string>("outInTrackCandidateSCBarrelAssociationCollection");
  InOutTrackSuperClusterBarrelAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateSCBarrelAssociationCollection");

  OutInTrackSuperClusterEndcapAssociationCollection_ = conf_.getParameter<std::string>("outInTrackCandidateSCEndcapAssociationCollection");
  InOutTrackSuperClusterEndcapAssociationCollection_ = conf_.getParameter<std::string>("inOutTrackCandidateSCEndcapAssociationCollection");

  // Register the product
  produces< TrackCandidateCollection > (OutInTrackCandidateBarrelCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateBarrelCollection_);
  produces< TrackCandidateCollection > (OutInTrackCandidateEndcapCollection_);
  produces< TrackCandidateCollection > (InOutTrackCandidateEndcapCollection_);
  //
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( OutInTrackSuperClusterBarrelAssociationCollection_);
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( InOutTrackSuperClusterBarrelAssociationCollection_);
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( OutInTrackSuperClusterEndcapAssociationCollection_);
  produces< reco::TrackCandidateSuperClusterAssociationCollection > ( InOutTrackSuperClusterEndcapAssociationCollection_);


}

ConversionTrackCandidateProducer::~ConversionTrackCandidateProducer() {


  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  
  delete theInOutTrackFinder_;
  delete theLayerMeasurements_;
  delete theNavigationSchool_;


}


void  ConversionTrackCandidateProducer::beginJob (edm::EventSetup const & theEventSetup) {
  nEvt_=0;
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
  nEvt_++;
  edm::LogInfo("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Analyzing event number: " << theEvent.id() << " Global Counter " << nEvt_ << "\n";
  LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Analyzing event number " <<   theEvent.id() <<  " Global Counter " << nEvt_ << "\n";
  
  
  // Update MeasurementTracker
  theMeasurementTracker_->update(theEvent);
  
  
  
  //
  // create empty output collections
  //
  //  Out In Track Candidates
  std::auto_ptr<TrackCandidateCollection> outInTrackCandidateBarrel_p(new TrackCandidateCollection); 
  std::auto_ptr<TrackCandidateCollection> outInTrackCandidateEndcap_p(new TrackCandidateCollection); 
  //  In Out  Track Candidates
  std::auto_ptr<TrackCandidateCollection> inOutTrackCandidateBarrel_p(new TrackCandidateCollection); 
  std::auto_ptr<TrackCandidateCollection> inOutTrackCandidateEndcap_p(new TrackCandidateCollection); 
  //   Track Candidate  Super Cluster Association
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> outInAssocBarrel_p(new reco::TrackCandidateSuperClusterAssociationCollection);
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> inOutAssocBarrel_p(new reco::TrackCandidateSuperClusterAssociationCollection);
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> outInAssocEndcap_p(new reco::TrackCandidateSuperClusterAssociationCollection);
  std::auto_ptr<reco::TrackCandidateSuperClusterAssociationCollection> inOutAssocEndcap_p(new reco::TrackCandidateSuperClusterAssociationCollection);


   
  // Get the basic cluster collection in the Barrel 
  edm::Handle<reco::BasicClusterCollection> bcBarrelHandle;
  theEvent.getByLabel(bcProducer_, bcBarrelCollection_, bcBarrelHandle);
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Trying to access basic cluster collection in the Barrel from my Producer " << "\n";
  reco::BasicClusterCollection clusterCollectionBarrel = *(bcBarrelHandle.product());
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer basic cluster collection size  " << clusterCollectionBarrel.size() << "\n";
  
  
  
  // Get the basic cluster collection in the Endcap 
  edm::Handle<reco::BasicClusterCollection> bcEndcapHandle;
  theEvent.getByLabel(bcProducer_, bcEndcapCollection_, bcEndcapHandle);
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Trying to access basic cluster collection in the Endcap from my Producer " << "\n";
  reco::BasicClusterCollection clusterCollectionEndcap = *(bcEndcapHandle.product());
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer basic cluster collection size  " << clusterCollectionEndcap.size() << "\n";
  
  
  // Get the Super Cluster collection in the Barrel
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Trying to access " << scHybridBarrelCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer barrel  SC collection size  " << scBarrelCollection.size() << "\n";
  
  // Get the Super Cluster collection in the Endcap
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Trying to access " <<scIslandEndcapCollection_.c_str() << "  from my Producer " << "\n";
  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Endcap SC collection size  " << scEndcapCollection.size() << "\n";
  

  /////////////


  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefBarrelForOutIn;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefBarrelForInOut;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefEndcapForOutIn;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefEndcapForInOut;  
  
  
  //  Loop over SC in the barrel and reconstruct converted photons
  int iSC=0; // index in product collection
  int lSC=0; // local index on barrel

  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scBarrelCollection.begin(); aClus != scBarrelCollection.end(); ++aClus) {
  
   LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() <<  " Energy " <<  aClus->energy() << "\n";

    theOutInSeedFinder_->setCandidate(*aClus);
    theOutInSeedFinder_->makeSeeds(  clusterCollectionBarrel );


    
    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidateBarrel_p);    

    theInOutSeedFinder_->setCandidate(*aClus);
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds(  clusterCollectionBarrel);
    
    std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidateBarrel_p); 


    // Debug
   LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size barrel " << (*outInTrackCandidateBarrel_p).size() << " in out track size " << (*inOutTrackCandidateBarrel_p).size() <<   "\n";


    //////////// Fill vectors of Ref to SC to be used for the Track-SC association
    reco::SuperClusterRef scRefOutIn(reco::SuperClusterRef(scBarrelHandle, lSC));
    reco::SuperClusterRef scRefInOut(reco::SuperClusterRef(scBarrelHandle, lSC));


    for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it !=  theOutInTracks.end(); ++it) {
      vecOfSCRefBarrelForOutIn.push_back( scRefOutIn );
           
     LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Barrel OutIn Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }

    for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it !=  theInOutTracks.end(); ++it) {
      vecOfSCRefBarrelForInOut.push_back( scRefInOut );

     LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Barrel InOut Tracks Number of hits " << (*it).foundHits() << "\n"; 
    }




    iSC++;
    lSC++;
  }


  //  Loop over SC in the Endcap and reconstruct tracks from converted photons
  lSC=0; // reset local index for endcap
  for(aClus = scEndcapCollection.begin(); aClus != scEndcapCollection.end(); ++aClus) {
  
     LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer SC eta " <<  aClus->eta() << " phi " <<  aClus->phi() << " Energy " <<  aClus->energy() << "\n";

    theOutInSeedFinder_->setCandidate(*aClus);
    theOutInSeedFinder_->makeSeeds(  clusterCollectionEndcap );

    std::vector<Trajectory> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds(),  *outInTrackCandidateEndcap_p);    
    
    theInOutSeedFinder_->setCandidate(*aClus);
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds(  clusterCollectionEndcap );
    LogDebug("ConversionTrackCandidateProducer")  << " ConversionTrackCandidateProducer inout seedfinding ended. Going to make the inout tracks " << "\n";
    
    
    std::vector<Trajectory> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds(),  *inOutTrackCandidateEndcap_p); 
    
    
  LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() <<  " Event pointer to out in track size endcap " << (*outInTrackCandidateEndcap_p).size() << " in out track size " << (*inOutTrackCandidateEndcap_p).size() <<   "\n";
  
  
  //////////// Fill vectors of Ref to SC to be used for the Track-SC association
  
  reco::SuperClusterRef scRefInOut(reco::SuperClusterRef(scEndcapHandle, lSC));
  reco::SuperClusterRef scRefOutIn(reco::SuperClusterRef(scEndcapHandle, lSC));
  
  for (std::vector<Trajectory>::const_iterator it = theOutInTracks.begin(); it !=  theOutInTracks.end(); ++it) {
    
    vecOfSCRefEndcapForOutIn.push_back( scRefOutIn );
    LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Endcap OutIn Tracks Number of hits " << (*it).foundHits() << "\n"; 
  }
  
  for (std::vector<Trajectory>::const_iterator it = theInOutTracks.begin(); it !=  theInOutTracks.end(); ++it) {
    
    vecOfSCRefEndcapForInOut.push_back( scRefInOut );
    LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Endcap InOut Tracks Number of hits " << (*it).foundHits() << "\n"; 
  }
  
    
    iSC++;
    lSC++; 
  }



  LogDebug("ConversionTrackCandidateProducer")  << "  ConversionTrackCandidateProducer vecOfSCRefBarrelForOutIn size " << vecOfSCRefBarrelForOutIn.size() << " vecOfSCRefBarrelForInOut size " << vecOfSCRefBarrelForInOut.size()  << "\n"; 
  LogDebug("ConversionTrackCandidateProducer")  << "  ConversionTrackCandidateProducer vecOfSCRefEndcapForOutIn size " << vecOfSCRefEndcapForOutIn.size() << " vecOfSCRefEndcapForInOut size " << vecOfSCRefEndcapForInOut.size()  << "\n"; 
  

  // put all products in the event
 // Barrel
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidateBarrel_p).size() << " Out In track Candidates " << "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of outInTrackCandidates: " <<  (*outInTrackCandidateBarrel_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackCBarrel = theEvent.put( outInTrackCandidateBarrel_p, OutInTrackCandidateBarrelCollection_ );
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  refprodOutInTrackCBarrel size  " <<  (*(refprodOutInTrackCBarrel.product())).size()  <<  "\n";
 //
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidateBarrel_p).size() << " In Out track Candidates " <<  "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of inOutTrackCandidates: " <<  (*inOutTrackCandidateBarrel_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackCBarrel = theEvent.put( inOutTrackCandidateBarrel_p, InOutTrackCandidateBarrelCollection_ );
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  refprodInOutTrackCBarrel size  " <<  (*(refprodInOutTrackCBarrel.product())).size()  <<  "\n";
 // Endcap
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event " << (*outInTrackCandidateEndcap_p).size() << " Out In track Candidates " << "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of outInTrackCandidates: " <<  (*outInTrackCandidateEndcap_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodOutInTrackCEndcap = theEvent.put( outInTrackCandidateEndcap_p, OutInTrackCandidateEndcapCollection_ );
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  refprodOutInTrackCEndcap size  " <<  (*(refprodOutInTrackCEndcap.product())).size()  <<  "\n";
 //
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event  " << (*inOutTrackCandidateEndcap_p).size() << " In Out track Candidates " <<  "\n";
 edm::LogInfo("ConversionTrackCandidateProducer") << "Number of inOutTrackCandidates: " <<  (*inOutTrackCandidateEndcap_p).size() << "\n";
 const edm::OrphanHandle<TrackCandidateCollection> refprodInOutTrackCEndcap = theEvent.put( inOutTrackCandidateEndcap_p, InOutTrackCandidateEndcapCollection_ );
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer  refprodInOutTrackCEndcap size  " <<  (*(refprodInOutTrackCEndcap.product())).size()  <<  "\n";


  

 LogDebug("ConversionTrackCandidateProducer") << " ConversionTrackCandidateProduce Going to fill association maps for the barrel " <<  "\n";
 for (unsigned int i=0;i< vecOfSCRefBarrelForOutIn.size(); ++i) {
   outInAssocBarrel_p->insert(edm::Ref<TrackCandidateCollection>(refprodOutInTrackCBarrel,i), vecOfSCRefBarrelForOutIn[i]  );
 }
 for (unsigned int i=0;i< vecOfSCRefBarrelForInOut.size(); ++i) {
   inOutAssocBarrel_p->insert(edm::Ref<TrackCandidateCollection>(refprodInOutTrackCBarrel,i), vecOfSCRefBarrelForInOut[i]  );
 }
 

 LogDebug("ConversionTrackCandidateProducer") << " ConversionTrackCandidateProduce Going to fill association maps for the endcap " <<  "\n";
 for (unsigned int i=0;i< vecOfSCRefEndcapForOutIn.size(); ++i) {
   outInAssocEndcap_p->insert(edm::Ref<TrackCandidateCollection>(refprodOutInTrackCEndcap,i), vecOfSCRefEndcapForOutIn[i]  );
 }
 for (unsigned int i=0;i< vecOfSCRefEndcapForInOut.size(); ++i) {
   inOutAssocEndcap_p->insert(edm::Ref<TrackCandidateCollection>(refprodInOutTrackCEndcap,i), vecOfSCRefEndcapForInOut[i]  );
 }
 
  
  
 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC Barrel association: size  " <<  (*outInAssocBarrel_p).size() << "\n";  
 theEvent.put( outInAssocBarrel_p, OutInTrackSuperClusterBarrelAssociationCollection_);
 
 LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC Barrel association: size  " <<  (*inOutAssocBarrel_p).size() << "\n";  
 theEvent.put( inOutAssocBarrel_p, InOutTrackSuperClusterBarrelAssociationCollection_);

 LogDebug("ConversionTrackCandidateProducer")  << "ConversionTrackCandidateProducer Putting in the event   OutIn track - SC Endcap association: size  " <<  (*outInAssocEndcap_p).size() << "\n";  
 theEvent.put( outInAssocEndcap_p, OutInTrackSuperClusterEndcapAssociationCollection_);
 
 LogDebug("ConversionTrackCandidateProducer") << "ConversionTrackCandidateProducer Putting in the event   InOut track - SC Endcap association: size  " <<  (*inOutAssocEndcap_p).size() << "\n";  
 theEvent.put( inOutAssocEndcap_p, InOutTrackSuperClusterEndcapAssociationCollection_);


  
}
