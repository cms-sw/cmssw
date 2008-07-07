/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: 2008/04/24 18:14:59 $
 *  $Revision: 1.43 $
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

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par, const MuonServiceProxy* service):theService(service) {
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Constructor called." << endl;

  theFitterName = par.getParameter<string>("FitterName");
  theNumberOfIterations = par.getParameter<unsigned int>("NumberOfIterations");
  isForceAllIterations = par.getParameter<bool>("ForceAllIterations");
}

/// Destructor
StandAloneMuonRefitter::~StandAloneMuonRefitter() {
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Destructor called." << endl;
}

  // Operations

/// Refit
StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::singleRefit(const Trajectory& trajectory) {
  
  theService->eventSetup().get<TrackingComponentsRecord>().get(theFitterName, theFitter);

  vector<Trajectory> refitted;

//   refitted = theFitter->fit(trajectory);                                         // old option 1

  TrajectoryMeasurement lastTM = trajectory.lastMeasurement();                                      // old option 3
  TrajectoryStateOnSurface firstTsos = TrajectoryStateWithArbitraryError()(lastTM.updatedState());  //
  TransientTrackingRecHit::ConstRecHitContainer trajRH = trajectory.recHits();                      //
  reverse(trajRH.begin(),trajRH.end());                                                             //
  refitted = theFitter->fit(trajectory.seed(), trajRH, firstTsos);                                  //

  if(!refitted.empty()) return RefitResult(true,refitted.front());
  else return RefitResult(false,trajectory);
}


StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::refit(const Trajectory& trajectory) {

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "---------------------------------" << endl;
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Starting refitting loop:" << endl;

  unsigned int nSuccess=0;
  Trajectory lastFitted=trajectory;
  bool allIter=true;

  for(unsigned int j=0; j<theNumberOfIterations; ++j) {

    StandAloneMuonRefitter::RefitResult singleRefitResult = singleRefit(lastFitted);
    lastFitted = singleRefitResult.second;

    if(!singleRefitResult.first) {
      allIter=false;
      LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "  refit n. " << nSuccess+1 << ": failed" << endl;
      break;
    }

    nSuccess++;
    LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "  refit n. " << nSuccess << ": OK" << endl;

  } // end for

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << nSuccess << " successful refits!" << endl;

  if(isForceAllIterations)
    return ( allIter ? RefitResult(true, lastFitted) : RefitResult(false, trajectory) );

  else return RefitResult(allIter, lastFitted);

}
