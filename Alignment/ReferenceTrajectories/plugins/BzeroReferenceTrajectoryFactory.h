#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm
namespace reco {
  class BeamSpot;
}

/// A factory that produces instances of class BzeroReferenceTrajectory from a
/// given TrajTrackPairCollection.

class BzeroReferenceTrajectoryFactory : public TrajectoryFactoryBase {
public:
  BzeroReferenceTrajectoryFactory(const edm::ParameterSet &config);
  ~BzeroReferenceTrajectoryFactory() override;

  /// Produce the reference trajectories.
  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
                                                   const ConstTrajTrackPairCollection &tracks,
                                                   const reco::BeamSpot &beamSpot) const override;

  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
                                                   const ConstTrajTrackPairCollection &tracks,
                                                   const ExternalPredictionCollection &external,
                                                   const reco::BeamSpot &beamSpot) const override;

  BzeroReferenceTrajectoryFactory *clone() const override { return new BzeroReferenceTrajectoryFactory(*this); }

private:
  double theMass;
  double theMomentumEstimate;
};
