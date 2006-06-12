/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date: 2006/06/06 17:08:33 $
 *   $Revision: 1.7 $
 *
 *   \author  R.Bellan - INFN TO
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"

#include "RecoMuon/StandAloneMuonProducer/src/StandAloneMuonProducer.h"

// TrackFinder and Specific STA Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

// Input and output collection

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
StandAloneMuonProducer::StandAloneMuonProducer(const ParameterSet& parameterSet){
  cout<<"StandAloneMuonProducer::StandAloneMuonProducer called"<<endl;

  // Parameter set for the Builder
  ParameterSet STA_pSet = parameterSet.getParameter<ParameterSet>("STATrajBuilderParameters");
  
  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<string>("MuonSeedCollectionLabel");

  // instantiate the concrete trajectory builder in the Track Finder
  // FIXME: potential memory leak??
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(STA_pSet));
  
  produces<reco::TrackCollection>();
}
  
/// destructor
StandAloneMuonProducer::~StandAloneMuonProducer(){
  cout<<"StandAloneMuonProducer destructor called"<<endl;
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void StandAloneMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  cout<<"StandAloneMuonProducer::produce"<<endl;

  // Take the seeds container
  cout<<"Taking the seeds"<<endl;
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel(theSeedCollectionLabel,seeds);

  // Percolate the event setup
  cout<<"Event setup percolation"<<endl;
  theTrackFinder->setES(eventSetup);

  // Percolate the event 
  cout<<"Event percolation"<<endl;
  theTrackFinder->setEvent(event);

  // Reconstruct 
  cout<<"Track Reconstruction"<<endl;
  std::auto_ptr<reco::TrackCollection> recMuons
    = theTrackFinder->reconstruct(seeds);

  // Load the RecMuon Container in the Event
  cout<<"Load the tracks in the event"<<endl;
  event.put(recMuons);
}

