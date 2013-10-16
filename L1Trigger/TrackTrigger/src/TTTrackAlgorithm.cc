/*! \brief   Implementation of methods of TTTrackAlgorithm
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Anders Ryd
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithm.h"

/// Fit the track
template< >
void TTTrackAlgorithm< Ref_PixelDigi_ >::FitTrack( TTTrack< Ref_PixelDigi_ > &seed ) const
{
  /// Get the Stubs and other information from the seed
  std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > curStubs = seed.getStubPtrs();
  GlobalVector seedMomentum = seed.getMomentum();
  GlobalPoint seedVertex = seed.getVertex();
  double seedRInv = seed.getRInv();
  double seedPhi0 = seedMomentum.phi();
  double seedZ0 = seedVertex.z();
  double seedCotTheta0 = tan( M_PI_2 - seedMomentum.theta() );

#include "L1Trigger/TrackTrigger/src/TTTrackAlgorithm_TrackFit.icc"

  seed.setMomentum( GlobalVector( newPt*cos(newPhi0),
                                  newPt*sin(newPhi0),
                                  newPt*newCotTheta0 ) );
  seed.setVertex( GlobalPoint( 0, 0, seedZ0 + dZ0 ) );
  seed.setRInv( newRInv );
  seed.setChi2( chiSquare );
}

