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
 *   $Date:  $
 *   $Revision:  $
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

#include "RecoMuon/L2MuonProducer/interface/L2MuonProducer.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

//FIXME??
//#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//FIXME
#include "DataFormats/MuonReco/interface/RecoMuonCollection.h"

//FIXME check well the includes

using namespace edm;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet STA_pSet = parameterSet.getParameter<ParameterSet>("STATrajBuilder");

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(STA_pSet));


  produces<RecoMuonCollection>();  //FIXME fancy-name  
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
  // ##### FIXME fancy-names  #####

  // Take the seeds container
  Handle<TrajectorySeedCollection> seeds; 
  event.getByLabel("MuonSeedsForL2",seeds);

  // Reconstruct 
  std::auto_ptr<RecoMuonCollection> recMuons
    = theTrackFinder->reconstruct(seeds,eventSetup);

  // the best would be 
  //   = theTrackFinder->reconstruct(seeds);
  
  // Load the RecMuon Container in the Event
  event.put(recMuons);
}

