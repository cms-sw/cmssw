/**  \class L3MuonProducer
 * 
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   \author  A. Everett - Purdue University
 */

// Framework
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
  l2MuonToken_ = consumes<reco::TrackCollection>(theL2CollectionLabel);
  l2MuonTrajToken_ = consumes<std::vector<Trajectory> >(theL2CollectionLabel.label());
  l2AssoMapToken_ = consumes<TrajTrackAssociationCollection>(theL2CollectionLabel.label());
  updatedL2AssoMapToken_ = consumes<reco::TrackToTrackMap>(theL2CollectionLabel.label());

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  ConsumesCollector iC = consumesCollector();
  
  // instantiate the concrete trajectory builder in the Track Finder
  MuonTrackLoader* mtl = new MuonTrackLoader(trackLoaderParameters,iC,theService);
  L3MuonTrajectoryBuilder* l3mtb = new L3MuonTrajectoryBuilder(trajectoryBuilderParameters, theService,iC);
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
  event.getByToken(l2MuonToken_,L2Muons);

  Handle<vector<Trajectory> > L2MuonsTraj;
  vector<MuonTrajectoryBuilder::TrackCand> L2TrackCands;


  event.getByToken(l2MuonTrajToken_, L2MuonsTraj);      

  edm::Handle<TrajTrackAssociationCollection> L2AssoMap;
  event.getByToken(l2AssoMapToken_,L2AssoMap);

  edm::Handle<reco::TrackToTrackMap> updatedL2AssoMap;
  event.getByToken(updatedL2AssoMapToken_,updatedL2AssoMap);


      
  for(TrajTrackAssociationCollection::const_iterator it = L2AssoMap->begin(); it != L2AssoMap->end(); ++it){	
    const Ref<vector<Trajectory> > traj = it->key;
    const reco::TrackRef tkRegular  = it->val;
    reco::TrackRef tkUpdated;
    reco::TrackToTrackMap::const_iterator iEnd;
    reco::TrackToTrackMap::const_iterator iii;
    if ( theL2CollectionLabel.instance() == "UpdatedAtVtx") {
      iEnd = updatedL2AssoMap->end();
      iii = updatedL2AssoMap->find(it->val);
      if (iii != iEnd ) tkUpdated = (*updatedL2AssoMap)[it->val] ;
    }
    
    const reco::TrackRef tk = ( tkUpdated.isNonnull() ) ? tkUpdated : tkRegular ;      
    
    MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand((Trajectory*)(0),tk);
    if( traj->isValid() ) L2Cand.first = &*traj ;
    L2TrackCands.push_back(L2Cand);
  }
  
  theTrackFinder->reconstruct(L2TrackCands, event);      
  
  LogTrace(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
