#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpotFwd.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

/// A factory that produces instances of class BzeroReferenceTrajectory from a
/// given TrajTrackPairCollection.

class BzeroReferenceTrajectoryFactory : public TrajectoryFactoryBase {
public:
  BzeroReferenceTrajectoryFactory(const edm::ParameterSet &config, edm::ConsumesCollector &iC);
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

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagFieldToken;

private:
  double theMass;
  double theMomentumEstimate;
};
