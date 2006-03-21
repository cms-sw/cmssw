/**  \class StandAloneMuonProducer
 * 
 *   StandAlone muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *   $Date:  $
 *   $Revision:  $
 *
 *   \author  R.Bellan - INFN TO
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Handle.h"

#include "RecoMuon/StandAloneMuonProducer/interface/StandAloneMuonProducer.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

//FIXME includes

//FIXME??
#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
//FIXME
#include "DataFormats/MuonReco/interface/RecoMuonCollection.h"


using namespace edm;

/// constructor with config
StandAloneMuonProducer::StandAloneMuonProducer(const ParameterSet& parameterSet){

  // Parameter set for the Builder
  ParameterSet STA_pSet = parameterSet.getParameter<ParameterSet>("STATrajBuilder");

  // instantiate the concrete trajectory builder in the Track Finder
  theTrackFinder = new MuonTrackFinder(new StandAloneMuonTrajectoryBuilder(STA_pSet));


  produces<RecoMuonCollection>();  //FIXME fancy-name  
}
  
/// destructor
StandAloneMuonProducer::~StandAloneMuonProducer(){
  if (theTrackFinder) delete theTrackFinder;
}


/// reconstruct muons
void StandAloneMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
  // ##### FIXME fancy-names  #####

  // Take the seeds container
  Handle<TrackingSeedCollection> seeds; 
  event.getByLabel("MuonSeedsForStandAlone",seeds);

  // Reconstruct 
  std::auto_ptr<RecoMuonCollection> recMuons
    = theTrackFinder->reconstruct(seeds,eventSetup);

  // the best would be 
  //   = theTrackFinder->reconstruct(seeds);
  
  // Load the RecMuon Container in the Event
  event.put(recMuons);
}

