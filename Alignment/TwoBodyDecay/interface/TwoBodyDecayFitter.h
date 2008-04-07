#ifndef Alignment_TwoBodyDecay_TwoBodyDecayFitter_h
#define Alignment_TwoBodyDecay_TwoBodyDecayFitter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayVirtualMeasurement.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayEstimator.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayLinearizationPointFinder.h"

/** /class TwoBodyDecayFitter
 *
 *  /author Edmund Widl
 */


class TwoBodyDecayFitter
{

public:

  TwoBodyDecayFitter( const edm::ParameterSet & config );

  TwoBodyDecayFitter( const edm::ParameterSet & config,
		      const LinearizationPointFinder* vf,
		      const TwoBodyDecayLinearizationPointFinder* lpf,
		      const TwoBodyDecayEstimator* est );

  virtual ~TwoBodyDecayFitter( void );

  virtual const TwoBodyDecay estimate( const std::vector< reco::TransientTrack >& tracks,
				       const TwoBodyDecayVirtualMeasurement& vm ) const;

  virtual const TwoBodyDecay estimate( const std::vector< reco::TransientTrack >& tracks,
				       const std::vector< TrajectoryStateOnSurface >& tsos,
				       const TwoBodyDecayVirtualMeasurement& vm ) const;

  inline const TwoBodyDecayLinearizationPointFinder* linearizationPointFinder( void ) const { return theLinPointFinder.operator->(); }
  inline const TwoBodyDecayEstimator* estimator( void ) const { return theEstimator.operator->(); }
  inline const LinearizationPointFinder* vertexFinder( void ) const { return theVertexFinder.operator->(); }

  virtual TwoBodyDecayFitter* clone( void ) const { return new TwoBodyDecayFitter( *this ); }

private:

  typedef PerigeeLinearizedTrackState::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;

  DeepCopyPointerByClone< const LinearizationPointFinder > theVertexFinder;
  DeepCopyPointerByClone< const TwoBodyDecayLinearizationPointFinder > theLinPointFinder;
  DeepCopyPointerByClone< const TwoBodyDecayEstimator > theEstimator;

  LinearizedTrackStateFactory theLinTrackStateFactory;

};

#endif
