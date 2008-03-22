/**  \class FastL3MuonProducer
 * 
 *   FAst L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon and a tracker track.
 *
 *   $Date: 2007/12/28 21:00:27 $
 *   $Revision: 1.1 $
 *   \author  Patrick Janot - CERN
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FastSimulation/Muons/plugins/FastL3MuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "FastSimulation/Muons/interface/FastL3MuonTrajectoryBuilder.h"
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

//
// constructor with config
//
FastL3MuonProducer::FastL3MuonProducer(const edm::ParameterSet& parameterSet) {

  LogTrace("FastL3MuonProducer") << "constructor called" << std::endl;

  // Parameter set for the Builder
  edm::ParameterSet trajectoryBuilderParameters = 
    parameterSet.getParameter<edm::ParameterSet>("L3TrajBuilderParameters");

  // L2 Muon Collection Label
  theL2CollectionLabel = 
    parameterSet.getParameter<edm::InputTag>("MuonCollectionLabel");
  updatedAtVtx = (theL2CollectionLabel.instance() == "UpdatedAtVtx");

  // L2 semi-persistent flag
  theL2TrajectoryFlag = 
    parameterSet.getParameter<bool>("MuonTrajectoryAvailable");

  // service parameters
  edm::ParameterSet serviceParameters = 
    parameterSet.getParameter<edm::ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  edm::ParameterSet trackLoaderParameters = 
    parameterSet.getParameter<edm::ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // instantiate the concrete trajectory builder in the Track Finder
  MuonTrackLoader* mtl = new MuonTrackLoader(trackLoaderParameters,theService);
  l3mtb = new FastL3MuonTrajectoryBuilder(trajectoryBuilderParameters, theService);
  theTrackFinder = new MuonTrackFinder(l3mtb, mtl);

  theL2SeededTkLabel = trackLoaderParameters.getUntrackedParameter<std::string>("MuonSeededTracksInstance",std::string());
  
  produces<reco::TrackCollection>(theL2SeededTkLabel);
  produces<TrackingRecHitCollection>(theL2SeededTkLabel);
  produces<reco::TrackExtraCollection>(theL2SeededTkLabel);
  produces<std::vector<Trajectory> >(theL2SeededTkLabel) ;
  produces<TrajTrackAssociationCollection>(theL2SeededTkLabel);

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();

  produces<reco::MuonTrackLinksCollection>();

}


//
// destructor
//
FastL3MuonProducer::~FastL3MuonProducer() {

  LogTrace("FastL3MuonProducer") << "destructor called" << std::endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void FastL3MuonProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // const std::string metname = "Muon|RecoMuon|FastL3MuonProducer";  
  // LogTrace(metname)<<endl<<endl<<endl;
  // LogTrace(metname)<<"Fast L3 Muon Reconstruction started"<<endl;  

  typedef std::vector<Trajectory> TrajColl;

  // Update the services
  theService->update(eventSetup);

  // Take the L2 muon container(s)
  // LogTrace(metname)<<"Taking the L2 Muons "<<theL2CollectionLabel<<endl;

  edm::Handle<reco::TrackCollection> L2Muons;
  event.getByLabel(theL2CollectionLabel,L2Muons);

  edm::Handle<std::vector<Trajectory> > L2MuonsTraj;
  std::vector<MuonTrajectoryBuilder::TrackCand> L2TrackCands;

  if ( theL2TrajectoryFlag ) {

    event.getByLabel(theL2CollectionLabel.label(), L2MuonsTraj);      
    
    edm::Handle<TrajTrackAssociationCollection> L2AssoMap;
    event.getByLabel(theL2CollectionLabel.label(),L2AssoMap);

    edm::Handle<reco::TrackToTrackMap> updatedL2AssoMap;
    event.getByLabel(theL2CollectionLabel.label(),updatedL2AssoMap);
    
    TrajTrackAssociationCollection::const_iterator it = L2AssoMap->begin();
    TrajTrackAssociationCollection::const_iterator lt = L2AssoMap->end();
    for( ; it!=lt; ++it){	

      const edm::Ref<std::vector<Trajectory> > traj = it->key;
      const reco::TrackRef tk = updatedAtVtx ? (*updatedL2AssoMap)[it->val] : it->val;
      MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand(0,tk);
      if( traj->isValid() ) L2Cand.first = &*traj ;
      L2TrackCands.push_back(L2Cand);

    }

  } else {

    unsigned nMuons = L2Muons->size();
    for ( unsigned int position = 0; position != nMuons; ++position ) {

      reco::TrackRef L2TrackRef(L2Muons,position);
      MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand(0,L2TrackRef);
      L2TrackCands.push_back(L2Cand); 

    }

  }

  theTrackFinder->reconstruct(L2TrackCands, event);      
  l3mtb->clear();
  
  /*
  LogTrace(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
  */
}
