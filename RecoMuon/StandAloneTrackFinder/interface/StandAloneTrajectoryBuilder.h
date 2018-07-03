#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"


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
  StandAloneMuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*,edm::ConsumesCollector& iC);

  /// Destructor
  ~StandAloneMuonTrajectoryBuilder() override;

  // Returns a vector of the reconstructed trajectories compatible with
  // the given seed.
  TrajectoryContainer trajectories(const TrajectorySeed&) override;

  /// dummy implementation, unused in this class
  CandidateContainer trajectories(const TrackCand&) override {return CandidateContainer();}

  /// pre-filter
  StandAloneMuonFilter* filter() const {return theFilter;}

  /// actual filter
  StandAloneMuonFilter* bwfilter() const {return theBWFilter;}

  /// refitter of the hits container
  StandAloneMuonRefitter* refitter() const {return theRefitter;}

  /// Pass the Event to the algo at each event
  void setEvent(const edm::Event& event) override;
  
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
