/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2006/08/15 10:58:40 $
 *   $Revision: 1.14 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/StandAloneMuonProducer/src/StandAloneMuonProducer.h"

// TrackFinder and Specific STA Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

// Input and output collection

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
StandAloneMuonProducer::StandAloneMuonProducer(const ParameterSet& parameterSet){
  LogDebug("Muon|RecoMuon|StandAloneMuonProducer")<<"constructor called"<<endl;

  // Parameter set for the Builder
  ParameterSet STA_pSet = parameterSet.getParameter<ParameterSet>("STATrajBuilderParameters");
  
  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getUntrackedParameter<string>("MuonSeedCollectionLabel");

  // the services
  theService = new MuonServiceProxy(parameterSet);

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(STA_pSet,theService));
  
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}
  
/// destructor
StandAloneMuonProducer::~StandAloneMuonProducer(){
  LogDebug("Muon|RecoMuon|StandAloneMuonProducer")<<"StandAloneMuonProducer destructor called"<<endl;
  if (theService) delete theService;
  if (theTrackFinder) delete theTrackFinder;
}

/// reconstruct muons
void StandAloneMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  const std::string metname = "Muon|RecoMuon|StandAloneMuonProducer";
  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"Stand Alone Muon Reconstruction Started"<<endl;

  // Take the seeds container
  LogDebug(metname)<<"Taking the seeds: "<<theSeedCollectionLabel<<endl;
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Update the services
  theService->update(eventSetup);

  // Reconstruct 
  LogDebug(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(seeds,event,eventSetup);
 
  LogDebug(metname)<<"Event loaded"
		   <<"================================"
		   <<endl<<endl;
}

