#ifndef Alignment_ReferenceTrajectories_CosmicsReferenceTrajectoryFactory_h
#define Alignment_ReferenceTrajectories_CosmicsReferenceTrajectoryFactory_h

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/CosmicsReferenceTrajectory.h"

/// A factory that produces instances of class CosmicsReferenceTrajectory from a given TrajTrackPairCollection.


class CosmicsReferenceTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  CosmicsReferenceTrajectoryFactory( const edm::ParameterSet & config );
  virtual ~CosmicsReferenceTrajectoryFactory( void );

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const;

  virtual CosmicsReferenceTrajectoryFactory* clone( void ) const { return new CosmicsReferenceTrajectoryFactory( *this ); }

private:

  double theMass;
  double theMomentumEstimate;
};


#endif
