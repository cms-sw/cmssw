/**  \class L3MuonProducer
 * 
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   $Date: 2007/08/22 17:44:31 $
 *   $Revision: 1.10 $
 *   \author  A. Everett - Purdue University
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/L3TrackFinder/interface/L3MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

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
L3MuonProducer::L3MuonProducer(const ParameterSet& parameterSet) {

  LogTrace("L3MuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("L3TrajBuilderParameters");

  // L2 Muon Collection Label
  theL2CollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");

  // L2 semi-persistent flag
  theL2TrajectoryFlag = parameterSet.getParameter<bool>("MuonTrajectoryAvailable");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // instantiate the concrete trajectory builder in the Track Finder
  MuonTrackLoader* mtl = new MuonTrackLoader(trackLoaderParameters,theService);
  L3MuonTrajectoryBuilder* l3mtb = new L3MuonTrajectoryBuilder(trajectoryBuilderParameters, theService);
  theTrackFinder = new MuonTrackFinder(l3mtb, mtl);

  theL2SeededTkLabel = trackLoaderParameters.getUntrackedParameter<std::string>("MuonSeededTracksInstance",std::string());
  
  produces<reco::TrackCollection>(theL2SeededTkLabel);
  produces<TrackingRecHitCollection>(theL2SeededTkLabel);
  produces<reco::TrackExtraCollection>(theL2SeededTkLabel);
  produces<vector<Trajectory> >(theL2SeededTkLabel) ;
  produces<TrajTrackAssociationCollection>(theL2SeededTkLabel);

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();

  produces<reco::MuonTrackLinksCollection>();

}


//
// destructor
//
L3MuonProducer::~L3MuonProducer() {

  LogTrace("L3MuonProducer") << "destructor called" << endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void L3MuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|L3MuonProducer";  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"L3 Muon Reconstruction started"<<endl;  

  typedef vector<Trajectory> TrajColl;

  // Update the services
  theService->update(eventSetup);

  // Take the L2 muon container(s)
  LogTrace(metname)<<"Taking the L2 Muons "<<theL2CollectionLabel<<endl;

  Handle<reco::TrackCollection> L2Muons;
  event.getByLabel(theL2CollectionLabel,L2Muons);

  Handle<vector<Trajectory> > L2MuonsTraj;
  vector<MuonTrajectoryBuilder::TrackCand> L2TrackCands;

  if(theL2TrajectoryFlag) {
    event.getByLabel(theL2CollectionLabel.label(), L2MuonsTraj);      
    
    edm::Handle<TrajTrackAssociationCollection> L2AssoMap;
    event.getByLabel(theL2CollectionLabel.label(),L2AssoMap);

    edm::Handle<reco::TrackToTrackMap> updatedL2AssoMap;
    event.getByLabel(theL2CollectionLabel.label(),updatedL2AssoMap);
    
    for(TrajTrackAssociationCollection::const_iterator it = L2AssoMap->begin(); it != L2AssoMap->end(); ++it){	
      const Ref<vector<Trajectory> > traj = it->key;
      const reco::TrackRef tk = (theL2CollectionLabel.instance() == "UpdatedAtVtx") ? (*updatedL2AssoMap)[it->val] : it->val;
      MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand(0,tk);
      if( traj->isValid() ) L2Cand.first = &*traj ;
      L2TrackCands.push_back(L2Cand);
    }
  } else {
    for ( unsigned int position = 0; position != L2Muons->size(); ++position ) {
      reco::TrackRef L2TrackRef(L2Muons,position);
      MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand(0,L2TrackRef);
      L2TrackCands.push_back(L2Cand); 
    }
  }

  theTrackFinder->reconstruct(L2TrackCands, event);      
  
  LogTrace(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
