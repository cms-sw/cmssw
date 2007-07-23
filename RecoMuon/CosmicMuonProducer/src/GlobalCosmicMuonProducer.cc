#include "RecoMuon/CosmicMuonProducer/src/GlobalCosmicMuonProducer.h"

/**\class GlobalCosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon
 *  tracks
 *
 * $Date: 2006/10/24 09:41:06 $
 * $Revision: 1.3 $
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
  
  // TrackLoader parameters
  edm::ParameterSet trackLoaderParameters = iConfig.getParameter<edm::ParameterSet>("TrackLoaderParameters");
  
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  theTrackFinder = new MuonTrackFinder(new GlobalCosmicMuonTrajectoryBuilder(tbpar,theService),
				       new MuonTrackLoader(trackLoaderParameters, theService));

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

  // The trajectory can be taken from the event!
  edm::Handle<std::vector<Trajectory> > staMuonsTraj;
  theTrackFinder->reconstruct(cosMuons,staMuonsTraj,iEvent);
  
  LogDebug(metname)<<"Event loaded";

}

