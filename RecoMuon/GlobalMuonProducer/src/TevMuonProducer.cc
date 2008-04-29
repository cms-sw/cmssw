/**  \class TevMuonProducer
 * 
 *   TeV muon reconstructor:
 *
 *
 *   $Date: 2008/02/27 21:50:41 $
 *   $Revision: 1.1 $
 *
 *   \author  Piotr Traczyk (SINS Warsaw)
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/GlobalMuonProducer/src/TevMuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"

// Input and output collection
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

using namespace edm;
using namespace std;

//
// constructor with config
//
TevMuonProducer::TevMuonProducer(const ParameterSet& parameterSet) {

  LogDebug("Muon|RecoMuon|TevMuonProducer") << "constructor called" << endl;

  // GLB Muon Collection Label
  theGLBCollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // TrackRefitter parameters
  ParameterSet refitterParameters = parameterSet.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new GlobalMuonRefitter(refitterParameters, theService);

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  theTrackLoader = new MuonTrackLoader(trackLoaderParameters,theService);

  theCocktails = parameterSet.getParameter< std::vector<std::string> >("Cocktails");
  theCocktailIndex = parameterSet.getParameter< std::vector<int> >("CocktailIndex");

  for(unsigned int ww=0;ww<theCocktails.size();ww++){
    LogDebug("Muon|RecoMuon|TevMuonProducer") << "Cocktail " << theCocktails[ww];
    //  setAlias(parameterSet.getParameter<std::string>("@module_label"));
    //  produces<int>();
    produces<reco::TrackCollection>(theCocktails[ww]);
    produces<TrackingRecHitCollection>(theCocktails[ww]);
    produces<reco::TrackExtraCollection>(theCocktails[ww]);
    produces<vector<Trajectory> >(theCocktails[ww]) ;
    produces<TrajTrackAssociationCollection>(theCocktails[ww]);
    produces<reco::TrackToTrackMap>(theCocktails[ww]);
    //  produces<reco::MuonTrackLinksCollection>().setBranchAlias(theAlias + "s");
  }
}


//
// destructor
//
TevMuonProducer::~TevMuonProducer() {

  LogTrace("Muon|RecoMuon|TevMuonProducer") << "destructor called" << endl;
  if (theService) delete theService;
  if (theRefitter) delete theRefitter;

}


//
// reconstruct muons
//
void TevMuonProducer::produce(Event& event, const EventSetup& eventSetup) {

  const string metname = "Muon|RecoMuon|TevMuonProducer";  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"TeV Muon Reconstruction started"<<endl;  

  // Update the services
  theService->update(eventSetup);

  theRefitter->setEvent(event);

  theRefitter->setServices(theService->eventSetup());

  // Take the GLB muon container(s)
  Handle<reco::TrackCollection> glbMuons;
  event.getByLabel(theGLBCollectionLabel,glbMuons);

  Handle<vector<Trajectory> > glbMuonsTraj;

  LogTrace(metname)<< "Taking " << glbMuons->size() << " Global Muons "<<theGLBCollectionLabel<<endl;

  vector<MuonTrajectoryBuilder::TrackCand> glbTrackCands;

  event.getByLabel(theGLBCollectionLabel.label(), glbMuonsTraj);
    
  const reco::TrackCollection *glbTracks = glbMuons.product();
  //  vector<Trajectory*> trajectories;
  Trajectory refitted;
  
  reco::TrackRef::key_type trackIndex = 0;
  for(unsigned int ww=0;ww<theCocktails.size();ww++){
    LogDebug(metname)<<"TeVRefit for cocktail: " <<theCocktailIndex[ww];
    std::vector<std::pair<Trajectory*,reco::TrackRef> > miniMap;
    vector<Trajectory*> trajectories;
    for (reco::TrackCollection::const_iterator track = glbTracks->begin(); track!=glbTracks->end(); track++ , ++trackIndex) {
      reco::TrackRef glbRef(glbMuons,trackIndex);
      refitted=theRefitter->refit(*track,theCocktailIndex[ww]);
      Trajectory *refit = new Trajectory(refitted);
      if (refitted.isValid()) {
	LogDebug(metname)<<"TeVTrackLoader for cocktail: " <<theCocktails[ww];
	trajectories.push_back(refit);
	std::pair<Trajectory*,reco::TrackRef> thisPair(refit,glbRef);
	miniMap.push_back(thisPair);
      }
    }
    theTrackLoader->loadTracks(trajectories,event,miniMap,theCocktails[ww]);
  }
    
  LogTrace(metname) << "Done." << endl;    

//  int output = 1;
//  std::auto_ptr< int > output_decision( new int(output) );
//  event.put(output_decision);
}
