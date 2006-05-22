/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/19 15:24:36 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include <vector>

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par){

}

void StandAloneMuonRefitter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastUpdatedTSOS =  theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  // FIXME
  // theNavLayers = vector<const DetLayer*>() ;
}

void StandAloneMuonRefitter::setES(const EventSetup& setup){}

void StandAloneMuonRefitter::refit(FreeTrajectoryState& initialFts){
  
  std::string metname = "StandAloneMuonRefitter::refit";
  bool timing = true;
  
  MuonPatternRecoDumper debug;
  LogDebug(metname) << "Starting the refit"; 
  TimeMe t(metname,timing);
  
  // this is the most outward FTS updated with a recHit
  FreeTrajectoryState lastUpdatedFts;
  // this is the last but one most outward FTS updated with a recHit
  FreeTrajectoryState lastButOneUpdatedFts;
  // this is the most outward FTS (updated or predicted)
  FreeTrajectoryState lastFts;
  
  lastUpdatedFts = lastButOneUpdatedFts = lastFts = initialFts;

  /*

  // FIXME: check the prop direction!
  vector<const DetLayer*> nLayers = navigation().compatibleLayers(infts,alongMomentum);  

  // FIXME: is it right?Or have I to invert the iterator/prop direction??
  vector<const DetLayer*>::iterator layer;

  // FIXME: begin() in rbegin() and end() in rend()??
  for ( layer = nLayers.begin(); layer!= nLayers.end(); ++layer ) {

    //    const DetLayer* layer = *nextlayer;
    debug.dumpLayer(layer,metname);

    LogDebug(metname) << "search TM from: " << lastFts.position();

    vector<TrajectoryMeasurement> measL = 
      theMeasureExtractor.measurements(*layer,,);
      layer->measurements(lastFts, propagator(), estimator());
    LogDebug(metname) << "MuonFTSRefiner: measL " << measL.size() << endl;
  */

}

