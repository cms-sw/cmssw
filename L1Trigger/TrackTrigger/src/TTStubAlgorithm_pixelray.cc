/*! \brief   Implementation of methods of TTStubAlgorithm_pixelray
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Kristofer Henriksson
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_pixelray.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

/// Matching operations
template< >
void TTStubAlgorithm_pixelray< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation, 
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

  /// Prepare pixelray
  std::pair< double, double >* rayEndpoints;

  /// Just call the helper function to do all the work
  rayEndpoints = this->GetPixelRayEndpoints( aTTStub,
                                             TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker,
                                             mCompatibilityScalingFactor );

  /// If positive define window
  if ( rayEndpoints )
  {
    /// Establish the valid window
    double positiveZBoundary =  mIPWidth/2;
    double negativeZBoundary = -mIPWidth/2;

    /// Is it really within the window?
    if ( ( rayEndpoints->second > negativeZBoundary ) &&
         ( rayEndpoints->first < positiveZBoundary ) )
    {
      aConfirmation = true;

      /// Calculate output
      /// NOTE this assumes equal pitch in both sensors!
      MeasurementPoint mp0 = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
      MeasurementPoint mp1 = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();
      aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

      /// By default, assigned as ZERO
      anOffset = 0;

      delete rayEndpoints;
    }
  }
  else
    delete rayEndpoints;

}

/// Get pixel ray end points
template< >
std::pair< double, double >* TTStubAlgorithm_pixelray< Ref_PixelDigi_ >::GetPixelRayEndpoints( const TTStub< Ref_PixelDigi_ > & aTTStub,
                                                                                               const StackedTrackerGeometry* stackedTracker,
                                                                                               double scalingFactor )
{
  /// Get the coordinates of the boundaries of the inner and outer pixels.
  /// Code adapted from Cluster::averagePosition
  const GeomDetUnit* innerDet = stackedTracker->idToDetUnit( aTTStub.getDetId(), 0 );
  const GeomDetUnit* outerDet = stackedTracker->idToDetUnit( aTTStub.getDetId(), 1 );

  MeasurementPoint innerAvg = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
  MeasurementPoint outerAvg = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();

  StackedTrackerDetId innerDetId( aTTStub.getClusterRef(0)->getDetId() );
  StackedTrackerDetId outerDetId( aTTStub.getClusterRef(1)->getDetId() );
  unsigned int innerStackMember = aTTStub.getClusterRef(0)->getStackMember();
  unsigned int outerStackMember = aTTStub.getClusterRef(1)->getStackMember();
  unsigned int innerStack = innerDetId.iLayer();
  unsigned int outerStack = outerDetId.iLayer();
  unsigned int innerLadderPhi = innerDetId.iPhi();
  unsigned int outerLadderPhi = outerDetId.iPhi();
  unsigned int innerLadderZ = innerDetId.iZ();
  unsigned int outerLadderZ = outerDetId.iZ();

  Measurement2DVector pixRtlOffset(0, 1);

  /// Find leftmost and rightmost pixels of projection
  GlobalPoint innerPixLeft  = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg + pixRtlOffset ));
  GlobalPoint innerPixRight = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg ));
  GlobalPoint outerPixLeft  = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg + pixRtlOffset ));
  GlobalPoint outerPixRight = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg ));
  GlobalPoint oldInnerPixLeft  = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg + pixRtlOffset ));
  GlobalPoint oldInnerPixRight = innerDet->toGlobal(innerDet->topology().localPosition( innerAvg ));
  GlobalPoint oldOuterPixLeft  = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg + pixRtlOffset ));
  GlobalPoint oldOuterPixRight = outerDet->toGlobal(outerDet->topology().localPosition( outerAvg ));

  bool swap = false;

  /// Cross check and swap left/right
  if ( outerPixLeft.perp() < innerPixLeft.perp() )
  {
    GlobalPoint temp = innerPixLeft;
    innerPixLeft = outerPixLeft;
    outerPixLeft = temp;

    temp = innerPixRight;
    innerPixRight = outerPixRight;
    outerPixRight = temp;
    swap = true;
  }

  /// Get useful quantities
  /// Left and right pixel boundaries differ only in z, have same r and \phi
  double outerPointRadius = outerPixLeft.perp();
  double innerPointRadius = innerPixLeft.perp();

  double outerPointPhi = outerPixLeft.phi();
  double innerPointPhi = innerPixLeft.phi();
  double outerPointEta = outerPixLeft.eta();
  double innerPointEta = innerPixLeft.eta();

  double outerPixLeftX = outerPixLeft.x();
  double innerPixLeftX = innerPixLeft.x();
  double outerPixLeftY = outerPixLeft.y();
  double innerPixLeftY = innerPixLeft.y();

  double outerPixRightX = outerPixRight.x();
  double innerPixRightX = innerPixRight.x();
  double outerPixRightY = outerPixRight.y();
  double innerPixRightY = innerPixRight.y();

  double oldOuterPointRadius = oldOuterPixLeft.perp();
  double oldInnerPointRadius = oldInnerPixLeft.perp();
  double oldOuterPointPhi = oldOuterPixLeft.phi();
  double oldInnerPointPhi = oldInnerPixLeft.phi();
  double oldOuterPointEta = oldOuterPixLeft.eta();
  double oldInnerPointEta = oldInnerPixLeft.eta();

  if (outerPointRadius <= innerPointRadius || innerPixLeft.z() >= innerPixRight.z() || outerPixLeft.z() >= outerPixRight.z() )
  {
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", VALUE BEFORE THE SWAP " << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", oldOuterPointRadius "<< oldOuterPointRadius << ", oldInnerPointRadius " << oldInnerPointRadius << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", oldOuterPointPhi "<< oldOuterPointPhi << ", oldInnerPointPhi " << oldInnerPointPhi << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", oldOuterPointEta "<< oldOuterPointEta << ", oldInnerPointEta " << oldInnerPointEta << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", oldInnerPixLeft.z() "<< oldInnerPixLeft.z() << ", oldInnerPixRight.z() " << oldInnerPixRight.z() << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", oldOuterPixLeft.z() "<< oldOuterPixLeft.z() << ", oldOuterPixRight.z() " << oldOuterPixRight.z() << std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  std::endl;
    if (swap) std::cout << std::cout.precision(10) <<  __LINE__ << ", VALUE AFTER THE SWAP " << std::endl;

    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPixLeftX " << outerPixLeftX  << ", innerPixLeftX "  << innerPixLeftX  << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPixLeftY " << outerPixLeftY  << ", innerPixLeftY "  << innerPixLeftY  << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPixRightX "<< outerPixRightX << ", innerPixRightX " << innerPixRightX << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPixRightY "<< outerPixRightY << ", innerPixRightY " << innerPixRightY << std::endl;

    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPointRadius "<< outerPointRadius << ", innerPointRadius " << innerPointRadius << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPointPhi "<< outerPointPhi << ", innerPointPhi " << innerPointPhi << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPointEta "<< outerPointEta << ", innerPointEta " << innerPointEta << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerPixLeft.z() "<< innerPixLeft.z() << ", innerPixRight.z() " << innerPixRight.z() << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", outerPixLeft.z() "<< outerPixLeft.z() << ", outerPixRight.z() " << outerPixRight.z() << std::endl;
    std::cout << std::cout.precision(10) <<  std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", CLUSTER VALUES " << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerAvg "<< innerAvg << ", outerAvg "<< outerAvg  << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerDetId "<< innerDetId << ", outerDetId " << outerDetId << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerStackMember "<< innerStackMember << ", outerStackMember " << outerStackMember << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerStack "<< innerStack << ", outerStack " << outerStack << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerLadderPhi " << innerLadderPhi << ", outerLadderPhi " << outerLadderPhi << std::endl;
    std::cout << std::cout.precision(10) <<  __LINE__ << ", innerLadderZ " << innerLadderZ << ", outerLadderZ " << outerLadderZ << std::endl;
  }

  assert(outerPointRadius >= innerPointRadius);
  assert(innerPixLeft.z() < innerPixRight.z());
  assert(outerPixLeft.z() < outerPixRight.z());

  /// Check for seed compatibility given a pt cut
  /// Threshold computed from radial location of hits
  double deltaRadius = outerPointRadius - innerPointRadius;
  double deltaPhiThreshold = deltaRadius * scalingFactor;

  /// Calculate angular displacement from hit phi locations
  /// and renormalize it, if needed
  double deltaPhi = outerPointPhi - innerPointPhi;
  if ( deltaPhi < 0 ) deltaPhi = -deltaPhi;
  if ( deltaPhi > M_PI ) deltaPhi = 2*M_PI - deltaPhi;

  /// Apply selection based on Pt
  if ( deltaPhi < deltaPhiThreshold )
  {
    /// Check for backprojection to beamline
    double outerPixLeftZ  = outerPixLeft.z();
    double outerPixRightZ = outerPixRight.z();
    double innerPixLeftZ  = innerPixLeft.z();
    double innerPixRightZ = innerPixRight.z();

    /// The projection factor relates distances on the detector to distances on the beam
    /// The location of the projected pixel ray boundary is then found
    double projectFactor = outerPointRadius / deltaRadius;
    double rightPixelRayZ = outerPixLeftZ + (innerPixRightZ - outerPixLeftZ) * projectFactor;
    double leftPixelRayZ = outerPixRightZ + (innerPixLeftZ - outerPixRightZ) * projectFactor;

    /// Return where ray ends
    std::pair< double, double >* rayPair = new std::pair< double, double >( leftPixelRayZ, rightPixelRayZ );
    return rayPair;
  }

  return 0;
}

