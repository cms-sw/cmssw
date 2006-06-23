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
//
//#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//  Abstract classes for the converion tracking components
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackFinder.h"
// Class header file
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonProducer.h"


#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionSeedFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/OutInConversionTrackFinder.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/InOutConversionTrackFinder.h"

ConvertedPhotonProducer::ConvertedPhotonProducer(const edm::ParameterSet& config) : 
  conf_(config), 
  theMeasurementTracker_(0), 
  theNavigationSchool_(0), 
  theOutInSeedFinder_(0), 
  theOutInTrackFinder_(0), 
  theInOutSeedFinder_(0),
  theInOutTrackFinder_(0),
  isInitialized(0)

{

  std::cout << " ConvertedPhotonProducer CTOR " << std::endl;

  // use onfiguration file to setup input/output collection names
 

  scProducer_       = conf_.getParameter<std::string>("scProducer");
  scCollection_     = conf_.getParameter<std::string>("scCollection");
 


  ConvertedPhotonCollection_ = conf_.getParameter<std::string>("convertedPhotonCollection");

  // Register the product
  produces< reco::ConvertedPhotonCollection >(ConvertedPhotonCollection_);



}

ConvertedPhotonProducer::~ConvertedPhotonProducer() {

  delete theMeasurementTracker_;
  delete theOutInSeedFinder_; 
  delete theOutInTrackFinder_;
  delete theInOutSeedFinder_;  

}


void  ConvertedPhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {

  //get magnetic field
  edm::LogInfo("ConvertedPhotonProducer") << "get magnetic field" << "\n";
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF_);  


  theEventSetup .get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker_ );


 


    // get the measurement tracker 
    //ParameterSet mt_params = conf_.getParameter<ParameterSet>("MeasurementTrackerParameters") ;
    theMeasurementTracker_ = new MeasurementTracker(theEventSetup, conf_);
    theLayerMeasurements_  = new LayerMeasurements(theMeasurementTracker_);
    theNavigationSchool_   = new SimpleNavigationSchool( &(*theGeomSearchTracker_)  , &(*theMF_));
    NavigationSetter setter( *theNavigationSchool_);


    // get the Out In Seed Finder  
    edm::LogInfo("ConvertedPhotonProducer") << "get the OutInSeedFinder" << "\n";
    theOutInSeedFinder_ = new OutInConversionSeedFinder ( &(*theMF_) ,  theMeasurementTracker_ );

    // get the Out In Track Finder
    edm::LogInfo("ConvertedPhotonProducer") << "get the OutInTrackFinder" << "\n";
    theOutInTrackFinder_ = new OutInConversionTrackFinder ( theEventSetup, conf_, &(*theMF_),  theMeasurementTracker_  );


    // get the In Out Seed Finder  
    edm::LogInfo("ConvertedPhotonProducer") << "get the InOutSeedFinder" << "\n";
    theInOutSeedFinder_ = new InOutConversionSeedFinder ( &(*theMF_) ,  theMeasurementTracker_  );


    // get the In Out Track Finder
    edm::LogInfo("ConvertedPhotonProducer") << "get the InOutTrackFinder" << "\n";
    theInOutTrackFinder_ = new InOutConversionTrackFinder ( theEventSetup, conf_, &(*theMF_),  theMeasurementTracker_  );



}


void ConvertedPhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  edm::LogInfo("ConvertedPhotonProducer") << "Analyzing event number: " << theEvent.id() << "\n";


  // Update MeasurementTracker
    theMeasurementTracker_->update(theEvent);



  //
  // create empty output collections
  //
  std::auto_ptr< reco::ConvertedPhotonCollection > outputConvPhotonCollection(new reco::ConvertedPhotonCollection);
  std::cout << " Created empty ConvertedPhotonCollection size " <<   std::endl;

  



  // Get the basic cluster collection
  edm::Handle<reco::BasicClusterCollection> bccHandle;
  try {
  theEvent.getByType(bccHandle);
  } catch ( cms::Exception& ex ) {
    LogError("ConvertedPhotonProducer") << "Error! can't get the Basic Cluster collection " << std::endl ;
  } 
  std::cout << " Trying to access basic cluster collection from my Producer " << std::endl;
  reco::BasicClusterCollection clusterCollection = *(bccHandle.product());
  std::cout << " basic cluster collection size  " << clusterCollection.size() << std::endl;


  // Get the Super Cluster collection
  Handle<reco::SuperClusterCollection> scHandle;
  try{  
    theEvent.getByLabel(scProducer_,scCollection_,scHandle);
  } catch ( cms::Exception& ex ) {
    LogError("ConvertedPhotonProducer") << "Error! can't get the SC " << scCollection_.c_str() ;
  } 
  std::cout << " Trying to access SC collection from my Producer " << std::endl;
  reco::SuperClusterCollection scCollection = *(scHandle.product());
  std::cout << " SC collection size  " << scCollection.size() << std::endl;

  




  reco::ConvertedPhotonCollection myConvPhotons; 
  const std::vector<TrajectorySeed> theOutInSeeds;

  //  Loop over SC and reconstruct converted photons
  int myCands=0;
  reco::SuperClusterCollection::iterator aClus;
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {
    theOutInSeedFinder_->setCandidate(*aClus);

    theOutInSeedFinder_->makeSeeds( *(bccHandle.product()) );
   
    //    std::vector<const TrajectoryMeasurement*> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds());     
    std::vector<const Trajectory*> theOutInTracks= theOutInTrackFinder_->tracks(theOutInSeedFinder_->seeds());     

    theInOutSeedFinder_->setCandidate(*aClus);
    theInOutSeedFinder_->setTracks(  theOutInTracks );   
    theInOutSeedFinder_->makeSeeds( *(bccHandle.product()) );

    //    std::vector<const TrajectoryMeasurement*> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds());     
    std::vector<const Trajectory*> theInOutTracks= theInOutTrackFinder_->tracks(theInOutSeedFinder_->seeds());     

    // Define candidates with tracks

    std::cout << "  ConvertedPhotonProducer theOutInTracks.size() " << theOutInTracks.size() << " theInOutTracks.size() " << theInOutTracks.size() << std::endl;

    if ( theOutInTracks.size() ||  theOutInTracks.size() ) {

      // Track cleaning
      // Track pairing
      // vertex finding
      // final candidate 

      reco::ConvertedPhoton newCandidate;
      outputConvPhotonCollection->push_back(newCandidate);
      myCands++;      


    }

    
  }


  // put the product in the event
  std::cout << " Put the ConvertedPhotonCollection " << myCands << "  candidates " << std::endl;
  theEvent.put( outputConvPhotonCollection, ConvertedPhotonCollection_);


}
