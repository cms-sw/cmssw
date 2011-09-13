#ifndef Alignment_TwoBodyDecay_TwoBodyDecayEstimator_h
#define Alignment_TwoBodyDecay_TwoBodyDecayEstimator_h

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayVirtualMeasurement.h"

#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h" 
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** Class  TwoBodyDecayEstimator estimates the decay parameters and the corresponding error
 *  matrix (see TwoBodyDecayParameter) from two linearized tracks, and an hypothesis for the
 *  primary particle's mass and width as well as the mass of the secondary particles. It utilizes
 *  a robust M-estimator.
 *
 *  /author Edmund Widl
 */

class TwoBodyDecayEstimator
{

public:

  typedef PerigeeLinearizedTrackState::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  TwoBodyDecayEstimator( const edm::ParameterSet & config );
  virtual ~TwoBodyDecayEstimator( void ) {}

  virtual TwoBodyDecay estimate( const std::vector< RefCountedLinearizedTrackState > & linTracks,
				 const TwoBodyDecayParameters & linearizationPoint,
				 const TwoBodyDecayVirtualMeasurement & vm ) const;

  inline int ndf( void ) const { return theNdf; }
  inline const AlgebraicVector& pulls( void ) const { return thePulls; }

  virtual TwoBodyDecayEstimator* clone( void ) const { return new TwoBodyDecayEstimator( *this ); }

protected:

  virtual bool constructMatrices( const std::vector< RefCountedLinearizedTrackState > & linTracks,
				  const TwoBodyDecayParameters & linearizationPoint,
				  const TwoBodyDecayVirtualMeasurement & vm,
				  AlgebraicVector & vecM, AlgebraicSymMatrix & matG, AlgebraicMatrix & matA ) const;

private:

  bool checkValues( const AlgebraicVector & vec ) const;

  double theRobustificationConstant;
  double theMaxIterDiff;
  int theMaxIterations;
  bool theUseInvariantMass;

  mutable int theNdf;
  mutable AlgebraicVector thePulls;

};


#endif
