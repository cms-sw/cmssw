#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}
namespace reco {
  class BeamSpot;
}

/// A factory that produces instances of class BzeroReferenceTrajectory from a
/// given TrajTrackPairCollection.

class BzeroReferenceTrajectoryFactory : public TrajectoryFactoryBase
{
public:
  BzeroReferenceTrajectoryFactory(const edm::ParameterSet &config);
  virtual ~BzeroReferenceTrajectoryFactory();

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const reco::BeamSpot &beamSpot) const;

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const;

  virtual BzeroReferenceTrajectoryFactory* clone() const { return new BzeroReferenceTrajectoryFactory( *this ); }

private:

  double theMass;
  double theMomentumEstimate;
};

