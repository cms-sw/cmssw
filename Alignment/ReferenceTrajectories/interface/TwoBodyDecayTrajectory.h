#ifndef Alignment_ReferenceTrajectories_TwoBodyDecayTrajectory_h
#define Alignment_ReferenceTrajectories_TwoBodyDecayTrajectory_h

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectoryState.h"


/**
   by Edmund Widl, see CMS NOTE-2007/032.
 */

namespace reco { class BeamSpot; }

class TwoBodyDecayTrajectory : public ReferenceTrajectoryBase
{

public:

  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
  typedef std::pair< ConstRecHitContainer, ConstRecHitContainer > ConstRecHitCollection;

  TwoBodyDecayTrajectory(const TwoBodyDecayTrajectoryState& tsos,
                         const ConstRecHitCollection& recHits,
                         const MagneticField* magField,
                         const reco::BeamSpot& beamSpot,
                         const ReferenceTrajectoryBase::Config& config);

  TwoBodyDecayTrajectory( void );

  ~TwoBodyDecayTrajectory( void ) {}

  virtual TwoBodyDecayTrajectory* clone( void ) const
    { return new TwoBodyDecayTrajectory( *this ); }

  /**Number of RecHits belonging to the first and second track.
   */
  inline const std::pair< int, int > numberOfRecHits( void ) { return theNumberOfRecHits; }

private:

  bool construct(const TwoBodyDecayTrajectoryState& state,
                 const ConstRecHitCollection& recHits,
                 const MagneticField* field,
                 const reco::BeamSpot& beamSpot);

  void constructTsosVecWithErrors( const ReferenceTrajectory& traj1,
				   const ReferenceTrajectory& traj2,
				   const MagneticField* field );

  void constructSingleTsosWithErrors( const TrajectoryStateOnSurface & tsos,
				      int iTsos,
				      const MagneticField* field );

  const MaterialEffects materialEffects_;
  const PropagationDirection propDir_;
  const bool useRefittedState_;
  const bool constructTsosWithErrors_;

  std::pair< int, int > theNumberOfRecHits;
};

#endif
