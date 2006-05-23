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
 *   $Date: 2006/05/19 15:23:20 $
 *   $Revision: 1.3 $
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

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

using namespace edm;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet L2_pSet = parameterSet.getParameter<ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<string>("MuonSeedCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(L2_pSet));


  produces<reco::TrackCollection>();
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
  // Take the seeds container
  // FIXME: Change in the L1 container!
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Percolate the event setup
  theTrackFinder->setES(eventSetup);

  // Percolate the event
  theTrackFinder->setEvent(event);
  
  // Reconstruct 
  std::auto_ptr<reco::TrackCollection> recMuons
    = theTrackFinder->reconstruct(seeds);
  
  // Load the RecMuon Container in the Event
  event.put(recMuons);
}

