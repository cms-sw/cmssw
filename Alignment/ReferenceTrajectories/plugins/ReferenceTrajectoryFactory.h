#ifndef Alignment_ReferenceTrajectories_ReferenceTrajectoryFactory_h
#define Alignment_ReferenceTrajectories_ReferenceTrajectoryFactory_h

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.


class ReferenceTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  ReferenceTrajectoryFactory( const edm::ParameterSet & config );
  virtual ~ReferenceTrajectoryFactory( void );

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const;

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup& setup,
							    const ConstTrajTrackPairCollection& tracks,
							    const ExternalPredictionCollection& external ) const;

  virtual ReferenceTrajectoryFactory* clone( void ) const { return new ReferenceTrajectoryFactory( *this ); }

protected:

  double theMass;
};


#endif
