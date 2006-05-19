/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/18 09:53:21 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par){

}

void StandAloneMuonRefitter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastFTS = theLastBut1FTS = FreeTrajectoryState();
  
  // FIXME
  // theNavLayers = vector<const DetLayer*>() ;
}

void StandAloneMuonRefitter::setES(const EventSetup& setup){}

void StandAloneMuonRefitter::refit(FreeTrajectoryState& initialState){

}

