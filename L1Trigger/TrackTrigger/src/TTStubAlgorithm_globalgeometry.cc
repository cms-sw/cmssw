/*! \brief   Implementation of methods of TTStubAlgorithm_globalgeometry
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_globalgeometry.h"

/// Matching operations
template< >
void TTStubAlgorithm_globalgeometry< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                              int &aDisplacement,
                                                                              int &anOffset,
                                                                              const TTStub< Ref_PixelDigi_ > &aTTStub ) const
{
  /// Convert DetId
  StackedTrackerDetId stDetId( aTTStub.getDetId() );

  /// Force this to be a BARREL-only algorithm
  if ( stDetId.isEndcap() )
  {
    aConfirmation = false;
    return;
  }

  /// Get average position of Clusters composing the Stub
  GlobalPoint innerHitPosition = (*TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker).findAverageGlobalPosition( aTTStub.getClusterRef(0).get() );
  GlobalPoint outerHitPosition = (*TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker).findAverageGlobalPosition( aTTStub.getClusterRef(1).get() );

  /// Get useful quantities
  double outerPointRadius = outerHitPosition.perp();
  double innerPointRadius = innerHitPosition.perp();
  double outerPointPhi = outerHitPosition.phi();
  double innerPointPhi = innerHitPosition.phi();

  /// Check for seed compatibility given a pt cut
  /// Threshold computed from radial location of hits
  double deltaRadius = outerPointRadius - innerPointRadius;
  double deltaPhiThreshold = deltaRadius * mCompatibilityScalingFactor;

  /// Calculate angular displacement from hit phi locations
  /// and renormalize it, if needed
  double deltaPhi = outerPointPhi - innerPointPhi;
  if ( deltaPhi < 0 ) deltaPhi = -deltaPhi;
  if ( deltaPhi > M_PI ) deltaPhi = 2*M_PI - deltaPhi;

  /// Apply selection based on Pt
  if ( deltaPhi < deltaPhiThreshold )
  {
    /// Check for backprojection to beamline
    double innerPointZ = innerHitPosition.z();
    double outerPointZ = outerHitPosition.z();
    double positiveZBoundary =  (mIPWidth - outerPointZ) * deltaRadius;
    double negativeZBoundary = -(mIPWidth + outerPointZ) * deltaRadius;
    double multipliedLocation = (innerPointZ - outerPointZ) * outerPointRadius;

    /// Apply selection based on backprojected Z
    if ( (multipliedLocation < positiveZBoundary) && (multipliedLocation > negativeZBoundary) )
    {
      aConfirmation = true;

      /// Calculate output
      /// NOTE this assumes equal pitch in both sensors!
      MeasurementPoint mp0 = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
      MeasurementPoint mp1 = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();
      aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

      /// By default, assigned as ZERO
      anOffset = 0;

    } /// End of selection based on Z
  } /// End of selection based on Pt
}

