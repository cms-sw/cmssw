#ifndef Alignment_ReferenceTrajectories_TwoParticleEventTrajectoryFactory_h
#define Alignment_ReferenceTrajectories_TwoParticleEventTrajectoryFactory_h

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectory.h"

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"


class TwoBodyDecayTrajectoryFactory : public TrajectoryFactoryBase
{

public:

  typedef TwoBodyDecayVirtualMeasurement VirtualMeasurement;
  typedef TwoBodyDecayTrajectoryState::TsosContainer TsosContainer;
  typedef TwoBodyDecayTrajectory::ConstRecHitCollection ConstRecHitCollection;

  TwoBodyDecayTrajectoryFactory( const edm::ParameterSet & config );
  ~TwoBodyDecayTrajectoryFactory( void ) {}

  /// Produce the trajectories.
  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup & setup,
							    const ConstTrajTrackPairCollection & tracks ) const;

  virtual const ReferenceTrajectoryCollection trajectories( const edm::EventSetup& setup,
							    const ConstTrajTrackPairCollection& tracks,
							    const ExternalPredictionCollection& external ) const;

  virtual TwoBodyDecayTrajectoryFactory* clone( void ) const { return new TwoBodyDecayTrajectoryFactory( *this ); }

protected:

  const ReferenceTrajectoryCollection constructTrajectories( const ConstTrajTrackPairCollection& tracks,
							     const TwoBodyDecay& tbd,
							     const MagneticField* magField,
							     bool setParameterErrors ) const;

  bool match( const TrajectoryStateOnSurface& state,
	      const TransientTrackingRecHit::ConstRecHitPointer& recHit ) const;
    
  void produceVirtualMeasurement( const edm::ParameterSet & config );

  VirtualMeasurement theVM;
  GlobalPoint theBeamSpot;
  GlobalError theBeamSpotError;

  TwoBodyDecayFitter* theFitter;

  double theNSigmaCutValue;
  bool theUseRefittedStateFlag;
  bool theConstructTsosWithErrorsFlag;

};


#endif
