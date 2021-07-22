#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  \authors R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *           D. Trocino - INFN Torino <daniele.trocino@to.infn.it>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm
class MuonServiceProxy;
class TrajectoryFitter;
class Trajectory;

class StandAloneMuonRefitter {
public:
  typedef std::pair<bool, Trajectory> RefitResult;

public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par, edm::ConsumesCollector col, const MuonServiceProxy* service);

  /// Destructor
  virtual ~StandAloneMuonRefitter();

  // Operations

  /// Refit
  RefitResult singleRefit(const Trajectory&);
  RefitResult refit(const Trajectory&);

protected:
private:
  const MuonServiceProxy* theService;
  const edm::ESGetToken<TrajectoryFitter, TrajectoryFitter::Record> theFitterToken;
  edm::ESHandle<TrajectoryFitter> theFitter;
  unsigned int theNumberOfIterations;
  bool isForceAllIterations;
  double theMaxFractionOfLostHits;
  double errorRescale;
};
#endif
