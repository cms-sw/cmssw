#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"


using namespace std;


TwoBodyDecayFitter::TwoBodyDecayFitter( const edm::ParameterSet & config ) :
  theVertexFinder( new DefaultLinearizationPointFinder() ),
  theLinPointFinder( new TwoBodyDecayLinearizationPointFinder( config ) ),
  theEstimator( new TwoBodyDecayEstimator( config ) ) {}


TwoBodyDecayFitter::TwoBodyDecayFitter( const edm::ParameterSet & config,
					const LinearizationPointFinder* vf,
					const TwoBodyDecayLinearizationPointFinder* lpf,
					const TwoBodyDecayEstimator* est ) :
  theVertexFinder( vf->clone() ),
  theLinPointFinder( lpf->clone() ),
  theEstimator( est->clone() ) {}


TwoBodyDecayFitter::~TwoBodyDecayFitter( void ) {}


const TwoBodyDecay TwoBodyDecayFitter::estimate( const vector< reco::TransientTrack >& tracks,
						 const TwoBodyDecayVirtualMeasurement& vm ) const
{
  // get geometrical linearization point
  GlobalPoint linVertex = theVertexFinder->getLinearizationPoint( tracks );

  // create linearized track states
  vector< RefCountedLinearizedTrackState > linTracks;
  linTracks.push_back( theLinTrackStateFactory.linearizedTrackState( linVertex, tracks[0] ) );
  linTracks.push_back( theLinTrackStateFactory.linearizedTrackState( linVertex, tracks[1] ) );

  // get full linearization point (geomatrical & kinematical)
  const TwoBodyDecayParameters linPoint = theLinPointFinder->getLinearizationPoint( linTracks, vm.primaryMass(), vm.secondaryMass() );

  // make the fit
  return theEstimator->estimate( linTracks, linPoint, vm );
}
