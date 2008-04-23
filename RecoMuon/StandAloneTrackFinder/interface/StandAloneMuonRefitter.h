#ifndef RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H
#define RecoMuon_StandAloneTrackFinder_StandAloneMuonRefitter_H

/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"

namespace edm {class ParameterSet;}
class MuonServiceProxy;

class StandAloneMuonRefitter {
 public:
  typedef std::pair<bool,Trajectory> RefitResult;

 public:
  /// Constructor
  StandAloneMuonRefitter(const edm::ParameterSet& par, const MuonServiceProxy* service);

  /// Destructor
  virtual ~StandAloneMuonRefitter();

  // Operations

  /// Refit
  RefitResult refit(const Trajectory&);

protected:

private:
  const MuonServiceProxy* theService;
};
#endif

