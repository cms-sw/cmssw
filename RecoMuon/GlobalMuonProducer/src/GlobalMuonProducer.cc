/**  \class GlobalMuonProducer
 * 
 *   Global muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a standalone reonstructed muon.
 *
 *   $Date: 2009/07/29 12:12:45 $
 *   $Revision: 1.35 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"
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
GlobalMuonProducer::GlobalMuonProducer(const ParameterSet& parameterSet) {

  LogTrace("Muon|RecoMuon|GlobalMuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("GLBTrajBuilderParameters");
  InputTag trackCollectionTag = parameterSet.getParameter<InputTag>("TrackerCollectionLabel");
  trajectoryBuilderParameters.addParameter<InputTag>("TrackerCollectionLabel",trackCollectionTag);
  
  // STA Muon Collection Label
  theSTACollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // instantiate the concrete trajectory builder in the Track Finder
  MuonTrackLoader* mtl = new MuonTrackLoader(trackLoaderParameters,theService);
  GlobalMuonTrajectoryBuilder* gmtb = new GlobalMuonTrajectoryBuilder(trajectoryBuilderParameters, theService);

  theTrackFinder = new MuonTrackFinder(gmtb, mtl);

  setAlias(parameterSet.getParameter<std::string>("@module_label"));
  produces<reco::TrackCollection>().setBranchAlias(theAlias + "Tracks");
  produces<TrackingRecHitCollection>().setBranchAlias(theAlias + "RecHits");
  produces<reco::TrackExtraCollection>().setBranchAlias(theAlias + "TrackExtras");
  produces<vector<Trajectory> >().setBranchAlias(theAlias + "Trajectories") ;
  produces<TrajTrackAssociationCollection>().setBranchAlias(theAlias + "TrajTrackMap");
  produces<reco::MuonTrackLinksCollection>().setBranchAlias(theAlias + "s");
}


//
// destructor
//
GlobalMuonProducer::~GlobalMuonProducer() {

  LogTrace("Muon|RecoMuon|GlobalMuonProducer") << "destructor called" << endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;

}


//
// reconstruct muons
//
void GlobalMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|GlobalMuonProducer";  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"Global Muon Reconstruction started"<<endl;  

  // Update the services
  theService->update(eventSetup);

  // Take the STA muon container(s)
  Handle<reco::TrackCollection> staMuons;
  event.getByLabel(theSTACollectionLabel,staMuons);

  Handle<vector<Trajectory> > staMuonsTraj;

  LogTrace(metname) << "Taking " << staMuons->size() << " Stand Alone Muons "<<theSTACollectionLabel<<endl;

  vector<MuonTrajectoryBuilder::TrackCand> staTrackCands;

  edm::Handle<TrajTrackAssociationCollection> staAssoMap;

  edm::Handle<reco::TrackToTrackMap> updatedStaAssoMap;

  if( event.getByLabel(theSTACollectionLabel.label(), staMuonsTraj) && event.getByLabel(theSTACollectionLabel.label(),staAssoMap) && event.getByLabel(theSTACollectionLabel.label(),updatedStaAssoMap) ) {    
    
    for(TrajTrackAssociationCollection::const_iterator it = staAssoMap->begin(); it != staAssoMap->end(); ++it){	
      const Ref<vector<Trajectory> > traj = it->key;
      const reco::TrackRef tkRegular  = it->val;
      reco::TrackRef tkUpdated;
      reco::TrackToTrackMap::const_iterator iEnd;
      reco::TrackToTrackMap::const_iterator iii;
      if ( theSTACollectionLabel.instance() == "UpdatedAtVtx") {
	iEnd = updatedStaAssoMap->end();
	iii = updatedStaAssoMap->find(it->val);
	if (iii != iEnd ) tkUpdated = (*updatedStaAssoMap)[it->val] ;
      }
      
      int etaFlip1 = ((tkUpdated.isNonnull() && tkRegular.isNonnull()) && ( (tkUpdated->eta() * tkRegular->eta() ) < 0)) ? -1 : 1; 
      
      const reco::TrackRef tk = ( tkUpdated.isNonnull() && etaFlip1==1 ) ? tkUpdated : tkRegular ;

      MuonTrajectoryBuilder::TrackCand tkCand = MuonTrajectoryBuilder::TrackCand((Trajectory*)(0),tk);
      if( traj->isValid() ) tkCand.first = &*traj ;
      staTrackCands.push_back(tkCand);
    }
  } else {
    for ( unsigned int position = 0; position != staMuons->size(); ++position ) {
      reco::TrackRef staTrackRef(staMuons,position);
      MuonTrajectoryBuilder::TrackCand staCand = MuonTrajectoryBuilder::TrackCand((Trajectory*)(0),staTrackRef);
      staTrackCands.push_back(staCand); 
    }
  }
    
  theTrackFinder->reconstruct(staTrackCands, event);      
  
  
  LogTrace(metname)<<"Event loaded"
                   <<"================================"
                   <<endl<<endl;
    
}
