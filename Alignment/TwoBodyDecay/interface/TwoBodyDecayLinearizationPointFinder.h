#ifndef Alignment_TwoBodyDecay_TwoBodyDecayLinearizationPointFinder_h
#define Alignment_TwoBodyDecay_TwoBodyDecayLinearizationPointFinder_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayParameters.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h" 

/** Class  TwoBodyDecayLinearizationPointFinder computes a rough estimate of the parameters
 *  of a decay. This serves as linearization point for TwoBodyDecayEstimator.
 *
 *  /author Edmund Widl
 */



class TwoBodyDecayLinearizationPointFinder
{

public:

  typedef PerigeeLinearizedTrackState::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  TwoBodyDecayLinearizationPointFinder( const edm::ParameterSet & config ) {}

  virtual ~TwoBodyDecayLinearizationPointFinder( void ) {}

  virtual const TwoBodyDecayParameters
  getLinearizationPoint( const std::vector< RefCountedLinearizedTrackState > & tracks,
			 const double primaryMass,
			 const double secondaryMass ) const;

  virtual TwoBodyDecayLinearizationPointFinder* clone( void ) const { return new TwoBodyDecayLinearizationPointFinder( *this ); }

};

#endif
