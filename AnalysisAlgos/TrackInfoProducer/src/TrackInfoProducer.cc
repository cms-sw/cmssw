#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/Common/interface/EDProduct.h" 
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"



TrackInfoProducer::TrackInfoProducer(const edm::ParameterSet& iConfig):
    conf_(iConfig),
    theAlgo_(iConfig),
    forwardPredictedStateTag_(iConfig.getParameter<std::string>( "forwardPredictedState" )),
    backwardPredictedStateTag_(iConfig.getParameter<std::string>( "backwardPredictedState" )),
    updatedStateTag_(iConfig.getParameter<std::string>( "updatedState" )),
    combinedStateTag_(iConfig.getParameter<std::string>( "combinedState" ))
{
  // forward predicted state 
  produces<reco::TrackInfoCollection>(forwardPredictedStateTag_);

  // backward predicted state 
  produces<reco::TrackInfoCollection>(backwardPredictedStateTag_);

  // backward predicted + forward predicted + measured hit position 
  produces<reco::TrackInfoCollection>(updatedStateTag_);

  // backward predicted + forward predicted
  produces<reco::TrackInfoCollection>(combinedStateTag_);
}


void TrackInfoProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  //
  // create empty output collections
  //
  std::auto_ptr<reco::TrackInfoCollection>    outputFwdColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputBwdColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputUpdatedColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputCombinedColl (new reco::TrackInfoCollection);

  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTracks");
  edm::InputTag RHTag = conf_.getParameter<edm::InputTag>("rechits");
  try{  
    edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
    theEvent.getByLabel(TkTag,TrajectoryCollection);
    edm::Handle<TrackingRecHitCollection> rechitscollection;
    theEvent.getByLabel(RHTag,rechitscollection);
    //
    //run the algorithm  
    //
    theAlgo_.run(TrajectoryCollection.product(),&rechitscollection,*outputFwdColl,*outputBwdColl,*outputUpdatedColl, *outputCombinedColl);
  } 
  catch (cms::Exception &e){ edm::LogInfo("TrackInfoProducer") << "cms::Exception caught!!!" << "\n" << e << "\n";}
  //
  //put everything in the event
  theEvent.put(outputFwdColl,forwardPredictedStateTag_ );
  theEvent.put(outputBwdColl,backwardPredictedStateTag_ );
  theEvent.put(outputUpdatedColl,updatedStateTag_ );
  theEvent.put(outputCombinedColl,combinedStateTag_ );

}




