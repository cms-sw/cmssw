#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2008/08/07 12:30:37 $
 *  $Revision: 1.24 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"


class TrajectorySeed;
class StandAloneMuonFilter;
class StandAloneMuonBackwardFilter;
class StandAloneMuonRefitter;
class MuonServiceProxy;
class SeedTransformer;

namespace edm {class ParameterSet;}

class StandAloneMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

 public:
  typedef std::pair<const DetLayer*,TrajectoryStateOnSurface> DetLayerWithState;    
  
 public:
  /// Constructor with Parameter set and MuonServiceProxy
  StandAloneMuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*);

  /// Destructor
  virtual ~StandAloneMuonTrajectoryBuilder();

  // Returns a vector of the reconstructed trajectories compatible with
  // the given seed.
  TrajectoryContainer trajectories(const TrajectorySeed&);

  /// dummy implementation, unused in this class
  virtual CandidateContainer trajectories(const TrackCand&) {return CandidateContainer();}

  /// pre-filter
  StandAloneMuonFilter* filter() const {return theFilter;}

  /// actual filter
  StandAloneMuonFilter* bwfilter() const {return theBWFilter;}

  /// refitter of the hits container
  StandAloneMuonRefitter* refitter() const {return theRefitter;}

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);
  
 protected:

 private:
  
  DetLayerWithState propagateTheSeedTSOS(TrajectoryStateOnSurface& aTSOS, DetId& aDetId);

 private:

  /// Navigation type
  /// "Direct","Standard"
  std::string theNavigationType;

  recoMuon::SeedPosition theSeedPosition;
  
  /// Propagator for the seed extrapolation
  std::string theSeedPropagatorName;
  
  StandAloneMuonFilter* theFilter;
  StandAloneMuonFilter* theBWFilter;
  // FIXME
  //  StandAloneMuonBackwardFilter* theBWFilter;
  StandAloneMuonRefitter* theRefitter;
  SeedTransformer* theSeedTransformer;

  bool doBackwardFilter;
  bool doRefit;
  bool doSeedRefit;
  std::string theBWSeedType;

  const MuonServiceProxy *theService;
};
#endif
