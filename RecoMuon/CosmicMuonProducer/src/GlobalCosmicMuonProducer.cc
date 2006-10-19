#include "RecoMuon/CosmicMuonProducer/src/GlobalCosmicMuonProducer.h"

/**\class GlobalCosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon
 *  tracks
 *
 * $Date: 2006/09/22 18:52:37 $
 * $Revision: 1.1 $
 * \author:  Chang Liu  - Purdue University <Chang.Liu@cern.ch>
**/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "RecoMuon/CosmicMuonProducer/interface/GlobalCosmicMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

//
// constructors and destructor
//
GlobalCosmicMuonProducer::GlobalCosmicMuonProducer(const edm::ParameterSet& iConfig)
{

  edm::ParameterSet tbpar = iConfig.getParameter<edm::ParameterSet>("TrajectoryBuilderParameters");
  theTrackCollectionLabel = iConfig.getParameter<edm::InputTag>("MuonCollectionLabel");

  // service parameters
  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  // the propagator name for the track loader
  std::string trackLoaderPropagatorName = iConfig.getParameter<std::string>("TrackLoaderPropagatorName");
  bool theTrajectoryFlag = iConfig.getUntrackedParameter<bool>("PutTrajectoryIntoEvent",false);
  
  theTrackFinder = new MuonTrackFinder(new GlobalCosmicMuonTrajectoryBuilder(tbpar,theService),
				       new MuonTrackLoader(trackLoaderPropagatorName,theTrajectoryFlag, theService));

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<std::vector<Trajectory> >();

  produces<reco::MuonCollection>();

}


GlobalCosmicMuonProducer::~GlobalCosmicMuonProducer()
{
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;
}


// ------------ method called to produce the data  ------------
void
GlobalCosmicMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  const std::string metname = "Muon|RecoMuon|GlobalCosmicMuonProducer";  
  LogDebug(metname)<<"Global Cosmic Muon Reconstruction started";  
  
  edm::Handle<reco::TrackCollection> cosMuons;
  iEvent.getByLabel(theTrackCollectionLabel,cosMuons);
  
  // Update the services
  theService->update(iSetup);
  
  // Reconstruct the tracks in the tracker+muon system
  LogDebug(metname)<<"Track Reconstruction";
  theTrackFinder->reconstruct(cosMuons,iEvent);
  
  LogDebug(metname)<<"Event loaded";

}

