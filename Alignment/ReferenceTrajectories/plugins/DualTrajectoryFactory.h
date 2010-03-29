#ifndef Alignment_ReferenceTrajectories_DualTrajectoryFactory_h
#define Alignment_ReferenceTrajectories_DualTrajectoryFactory_h

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.


class DualTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  DualTrajectoryFactory( const edm::ParameterSet & config );
  virtual ~DualTrajectoryFactory( void );

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const;

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup& setup,
							    const ConstTrajTrackPairCollection& tracks,
							    const ExternalPredictionCollection& external ) const;

  virtual DualTrajectoryFactory* clone( void ) const { return new DualTrajectoryFactory( *this ); }

protected:

  struct DualTrajectoryInput
  {
    TrajectoryStateOnSurface refTsos;
    TransientTrackingRecHit::ConstRecHitContainer fwdRecHits;
    TransientTrackingRecHit::ConstRecHitContainer bwdRecHits;
  };

  const DualTrajectoryInput referenceStateAndRecHits( const ConstTrajTrackPair& track ) const;

  const TrajectoryStateOnSurface propagateExternal( const TrajectoryStateOnSurface& external,
						    const Surface& surface,
						    const MagneticField* magField  ) const;

  double theMass;
};


#endif
