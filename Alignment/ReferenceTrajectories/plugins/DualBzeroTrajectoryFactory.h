#ifndef Alignment_ReferenceTrajectories_DualBzeroTrajectoryFactory_h
#define Alignment_ReferenceTrajectories_DualBzeroTrajectoryFactory_h

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.


class DualBzeroTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  DualBzeroTrajectoryFactory( const edm::ParameterSet & config );
  virtual ~DualBzeroTrajectoryFactory( void );

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const;

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup& setup,
							    const ConstTrajTrackPairCollection& tracks,
							    const ExternalPredictionCollection& external ) const;

  virtual DualBzeroTrajectoryFactory* clone( void ) const { return new DualBzeroTrajectoryFactory( *this ); }

protected:

  struct DualBzeroTrajectoryInput
  {
    TrajectoryStateOnSurface refTsos;
    TransientTrackingRecHit::ConstRecHitContainer fwdRecHits;
    TransientTrackingRecHit::ConstRecHitContainer bwdRecHits;
  };

  const DualBzeroTrajectoryInput referenceStateAndRecHits( const ConstTrajTrackPair& track ) const;

  const TrajectoryStateOnSurface propagateExternal( const TrajectoryStateOnSurface& external,
						    const Surface& surface,
						    const MagneticField* magField  ) const;

  double theMass;
  double theMomentumEstimate;

};


#endif
