/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: 2011/01/10 00:16:32 $
 *  $Revision: 1.51 $
 *  \authors R. Bellan - INFN Torino <riccardo.bellan@cern.ch>,
 *           D. Trocino - INFN Torino <daniele.trocino@to.infn.it>
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par, const MuonServiceProxy* service):theService(service) {
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Constructor called." << endl;

  theFitterName = par.getParameter<string>("FitterName");
  theNumberOfIterations = par.getParameter<unsigned int>("NumberOfIterations");
  isForceAllIterations = par.getParameter<bool>("ForceAllIterations");
  theMaxFractionOfLostHits = par.getParameter<double>("MaxFractionOfLostHits");
  errorRescale = par.getParameter<double>("RescaleError");
}

/// Destructor
StandAloneMuonRefitter::~StandAloneMuonRefitter() {
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Destructor called." << endl;
}

  // Operations

/// Refit
StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::singleRefit(const Trajectory& trajectory) {
  
  theService->eventSetup().get<TrajectoryFitter::Record>().get(theFitterName, theFitter);

  vector<Trajectory> refitted;

  TrajectoryMeasurement lastTM = trajectory.lastMeasurement();                                      

  TrajectoryStateOnSurface firstTsos(lastTM.updatedState());

  // Rescale errors before refit, not to bias the result
  firstTsos.rescaleError(errorRescale);

  TransientTrackingRecHit::ConstRecHitContainer trajRH = trajectory.recHits();                      
  reverse(trajRH.begin(),trajRH.end());                                                             
  refitted = theFitter->fit(trajectory.seed(), trajRH, firstTsos);                                  

  if(!refitted.empty()) return RefitResult(true,refitted.front());
  else return RefitResult(false,trajectory);
}


StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::refit(const Trajectory& trajectory) {

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "---------------------------------" << endl;
  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "Starting refitting loop:" << endl;

  unsigned int nSuccess=0;
  unsigned int nOrigHits=trajectory.recHits().size();
  Trajectory lastFitted=trajectory;
  bool allIter=true;
  bool enoughRH=true;

  for(unsigned int j=0; j<theNumberOfIterations; ++j) {

    StandAloneMuonRefitter::RefitResult singleRefitResult = singleRefit(lastFitted);
    lastFitted = singleRefitResult.second;
    unsigned int nLastHits=lastFitted.recHits().size();

    if(!singleRefitResult.first) {
      allIter=false;
      LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "  refit n. " << nSuccess+1 << ": failed" << endl;
      break;
    }

    double lostFract= 1 - double(nLastHits)/nOrigHits;
    if(lostFract>theMaxFractionOfLostHits) {
      enoughRH=false;
      LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "  refit n. " << nSuccess+1 << ": too many RH lost" << endl;
      LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "     Survived RecHits: " << nLastHits << "/" << nOrigHits << endl;
      break;
    }

    nSuccess++;
    LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "  refit n. " << nSuccess << ": OK" << endl;
    LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << "     Survived RecHits: " << nLastHits << "/" << nOrigHits << endl;

  } // end for

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter") << nSuccess << " successful refits!" << endl;

  // if isForceAllIterations==true  =>   3 successful refits: (true, refitted trajectory)
  //                                    <3 successful refits: (false, original trajectory)
  // if isForceAllIterations==false =>  >0 successful refits: (true, last refitted trajectory)
  //                                     0 successful refits: (false, original trajectory)
  if(!enoughRH)
    return RefitResult(false, trajectory);
  else if(isForceAllIterations)
    return allIter ? RefitResult(allIter, lastFitted) : RefitResult(allIter, trajectory);
  else
    return nSuccess==0 ? RefitResult(false, trajectory) : RefitResult(true, lastFitted);
}
