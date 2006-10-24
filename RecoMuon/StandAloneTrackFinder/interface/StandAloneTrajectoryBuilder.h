#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/10/05 13:21:48 $
 *  $Revision: 1.16 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"

class TrajectorySeed;
class StandAloneMuonRefitter;
class StandAloneMuonBackwardFilter;
class StandAloneMuonSmoother;
class MuonServiceProxy;

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

  // FIXME: not relevant here?
  virtual CandidateContainer trajectories(const reco::TrackRef&) {return CandidateContainer();}

  StandAloneMuonRefitter* refitter() const {return theRefitter;}
  //FIXME
  //  StandAloneMuonBackwardFilter* bwfilter() const {return theBWFilter;}
  StandAloneMuonRefitter* bwfilter() const {return theBWFilter;}
  StandAloneMuonSmoother* smoother() const {return theSmoother;}

  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);
  
 protected:

 private:
  
  DetLayerWithState propagateTheSeedTSOS(const TrajectorySeed& seed);

 private:

  /// Navigation type
  /// "Direct","Standard"
  std::string theNavigationType;

  recoMuon::SeedPosition theSeedPosition;
  
  /// Propagator for the seed extrapolation
  std::string theSeedPropagatorName;
  
  StandAloneMuonRefitter* theRefitter;
  StandAloneMuonRefitter* theBWFilter;
  // FIXME
  //  StandAloneMuonBackwardFilter* theBWFilter;
  StandAloneMuonSmoother* theSmoother;

  bool doBackwardRefit;
  bool doSmoothing;
  std::string theBWSeedType;

  const MuonServiceProxy *theService;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
};
#endif
