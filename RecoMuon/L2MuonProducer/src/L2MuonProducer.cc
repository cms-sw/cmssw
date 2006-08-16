//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   Level-2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from Level-1 trigger seeds.
 *
 *
 *   $Date: 2006/07/06 09:19:04 $
 *   $Revision: 1.8 $
 *
 *   \author  R.Bellan - INFN TO
 */
//
//--------------------------------------------------

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

using namespace edm;
using namespace std;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet L2_pSet = parameterSet.getParameter<ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<string>("MuonSeedCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder
  // FIXME: tmp!
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(L2_pSet));


  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
 const std::string metname = "Muon|RecoMuon|L2MuonProducer";
  
  LogDebug(metname)<<endl<<endl<<endl;
  LogDebug(metname)<<"L2 Muon Reconstruction Started"<<endl;
  
  // Take the seeds container
  // FIXME: Change in the L1 container!
  LogDebug(metname)<<"Taking the seeds: "<<theSeedCollectionLabel<<endl;
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Reconstruct 
  LogDebug(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(seeds,event,eventSetup);
  
  LogDebug(metname)<<"Event loaded"
		   <<"================================"
		   <<endl<<endl;
}

