/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: 2008/04/23 16:56:34 $
 *  $Revision: 1.42 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par, const MuonServiceProxy* service):theService(service){
  theFitterName = par.getParameter<string>("FitterName");
  theTEMPORARYoption = par.getParameter<int>("Option");
}

/// Destructor
StandAloneMuonRefitter::~StandAloneMuonRefitter(){

}

  // Operations

  /// Refit
StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::refit(const Trajectory& trajectory){
  
  theService->eventSetup().get<TrackingComponentsRecord>().get(theFitterName, theFitter);

  vector<Trajectory> refitted;

  if(theTEMPORARYoption == 1)
    refitted = theFitter->fit(trajectory);
  
  else if(theTEMPORARYoption == 2 || theTEMPORARYoption == 3){
    TrajectoryMeasurement lastTM = trajectory.lastMeasurement();
    TrajectoryStateOnSurface firstTsos = lastTM.updatedState();
    if (theTEMPORARYoption == 3) 
      firstTsos = TrajectoryStateWithArbitraryError()(lastTM.updatedState());
    TransientTrackingRecHit::ConstRecHitContainer trajRH = trajectory.recHits();
    reverse(trajRH.begin(),trajRH.end());

    vector<Trajectory> refitted = theFitter->fit(trajectory.seed(), trajRH, firstTsos);
  }
  
  if(!refitted.empty()) return RefitResult(true,refitted.front());
  else return RefitResult(false,trajectory);
}


// {
//   TransientTrackingRecHit::ConstRecHitContainer trajRH = trajectory.recHits();
//   for()

// }
