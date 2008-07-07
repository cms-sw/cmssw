#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: 2008/04/24 18:14:58 $
 *  $Revision: 1.30 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */


#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

namespace edm {class ParameterSet;}
class MuonServiceProxy;
class TrajectoryFitter;
class Trajectory;

class StandAloneMuonRefitter {
 public:
  typedef std::pair<bool, Trajectory> RefitResult;

 public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par, const MuonServiceProxy* service);

  /// Destructor
  virtual ~StandAloneMuonRefitter();

  // Operations

  /// Refit
  RefitResult singleRefit(const Trajectory&);
  RefitResult refit(const Trajectory&);

protected:

private:
  const MuonServiceProxy* theService;
  edm::ESHandle<TrajectoryFitter> theFitter;
  std::string  theFitterName;
  unsigned int theNumberOfIterations;
  bool isForceAllIterations;
};
#endif

