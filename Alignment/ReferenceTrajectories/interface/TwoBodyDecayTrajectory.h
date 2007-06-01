#ifndef Alignment_ReferenceTrajectories_TwoBodyDecayTrajectory_h
#define Alignment_ReferenceTrajectories_TwoBodyDecayTrajectory_h

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectoryState.h"


class TwoBodyDecayTrajectory : public ReferenceTrajectoryBase
{

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
  typedef std::pair< ConstRecHitContainer, ConstRecHitContainer > ConstRecHitCollection;

  TwoBodyDecayTrajectory( const TwoBodyDecayTrajectoryState & trajectoryState,
			  const ConstRecHitCollection & recHits,
			  const MagneticField* magField,
			  MaterialEffects materialEffects = combined,
			  bool hitsAreReverse = false,
			  bool useRefittedState = true,
			  bool constructTsosWithErrors = false );

  TwoBodyDecayTrajectory( void );

  ~TwoBodyDecayTrajectory( void ) {}

  virtual TwoBodyDecayTrajectory* clone( void ) const
    { return new TwoBodyDecayTrajectory( *this ); }

  /**Number of RecHits belonging to the first and second track.
   */
  inline const std::pair< int, int > numberOfRecHits( void ) { return theNumberOfRecHits; }

private:

  bool construct( const TwoBodyDecayTrajectoryState & state,
		  const ConstRecHitCollection & recHits,
		  const MagneticField* field,
		  MaterialEffects materialEffects,
		  bool useRefittedState,
		  bool constructTsosWithErrors );

  void constructTsosVecWithErrors( const ReferenceTrajectory& traj1,
				   const ReferenceTrajectory& traj2,
				   const MagneticField* field );

  void constructSingleTsosWithErrors( const TrajectoryStateOnSurface & tsos,
				      int iTsos,
				      const MagneticField* field );

  std::pair< int, int > theNumberOfRecHits;

};

#endif
