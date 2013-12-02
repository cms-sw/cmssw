/*! \brief   Implementation of methods of TTStubAlgorithm_window2012
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window2012.h"

/// Matching operations
template< >
void TTStubAlgorithm_window2012< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                          int &aDisplacement,
                                                                          int &anOffset,
                                                                          const TTStub< Ref_PixelDigi_ > &aTTStub ) const
{
  /// Calculate average coordinates col/row for inner/outer Cluster
  /// These are already corrected for being at the center of each pixel
  MeasurementPoint mp0 = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
  MeasurementPoint mp1 = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();

  /// Get the module position in global coordinates
  StackedTrackerDetId stDetId( aTTStub.getDetId() );
  const GeomDetUnit* det0 = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->idToDetUnit( stDetId, 0 );
  const GeomDetUnit* det1 = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->idToDetUnit( stDetId, 1 );

  /// Find pixel pitch and topology related information
  const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
  const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
  const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );
  std::pair< float, float > pitch0 = top0->pitch();
  std::pair< float, float > pitch1 = top1->pitch();

  /// Get the Stack radius and z and displacements
  double R0 = det0->position().perp();
  double R1 = det1->position().perp();
  double Z0 = det0->position().z();
  double Z1 = det1->position().z();

  double DR = R1-R0;
  double DZ = Z1-Z0;

  /// Scale factor is already present in
  /// double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;
  /// hence the formula iis something like
  /// displacement < Delta * 1 / sqrt( ( 1/(mPtScalingFactor*R) )** 2 - 1 )
  double denominator = sqrt( 1/( mPtScalingFactor*mPtScalingFactor*R0*R0 ) - 1 );

  if (stDetId.isBarrel())
  {
    /// All of these are calculated in terms of pixels in outer sensor
    /// 0) Calculate window in terms of multiples of outer sensor pitch
    int window = floor( (DR/denominator) / pitch1.first ) + 1;
    /// POSITION IN TERMS OF PITCH MULTIPLES:
    ///       0 1 2 3 4 5 5 6 8 9 ...
    /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
    /// OUT   | | | | | |x| | | | | | | | | |
    ///
    /// IN    | | | |x|x| | | | | | | | | | |
    ///             THIS is 3.5 (COORD) and 4.0 (POS)
    /// 1) disp is the difference between average row coordinates
    ///    in inner and outer stack member, in terms of outer member pitch
    ///    (in case they are the same, this is just a plain coordinate difference)
    double dispD = mp1.x() - mp0.x() * (pitch0.first / pitch1.first);
    int dispI = ((dispD>0)-(dispD<0))*floor(fabs(dispD));
    /// 2) offset is the projection with a straight line of the innermost
    ///    hit towards the ourermost stack member, still in terms of outer member pitch
    ///    NOTE: in terms of coordinates, the center of the module is at NROWS/2-0.5 to
    ///    be consistent with the definition given above 
    double offsetD = DR/R0 * ( mp0.x() - (top0->nrows()/2 - 0.5) ) * (pitch0.first / pitch1.first);
    int offsetI = ((offsetD>0)-(offsetD<0))*floor(fabs(offsetD));

    /// Accept the stub if the post-offset correction displacement is smaller than the half-window
    if ( abs(dispI - offsetI) < window )
    {
      aConfirmation = true;
      aDisplacement = 2*dispI; /// In HALF-STRIP units!
      anOffset = 2*offsetI; /// In HALF-STRIP units!
    } /// End of stub is accepted
  }
  else if (stDetId.isEndcap())
  {
    /// All of these are calculated in terms of pixels in outer sensor
    /// 0) Calculate window in terms of multiples of outer sensor pitch
    int window = floor( R0/Z0 * (DZ/denominator) / pitch1.first ) + 1;
    /// 1) disp is the difference between average row coordinates
    ///    in inner and outer stack member, in terms of outer member pitch
    ///    (in case they are the same, this is just a plain coordinate difference)
    double dispD = mp1.x() - mp0.x() * (pitch0.first / pitch1.first);
    int dispI = ((dispD>0)-(dispD<0))*floor(fabs(dispD));
    /// 2) offset is the projection with a straight line of the innermost
    ///    hit towards the ourermost stack member, still in terms of outer member pitch
    ///    NOTE: in terms of coordinates, the center of the module is at NROWS/2-0.5 to
    ///    be consistent with the definition given above 
    double offsetD = DZ/Z0 * ( mp0.x() - (top0->nrows()/2 - 0.5) ) * (pitch0.first / pitch1.first);
    int offsetI = ((offsetD>0)-(offsetD<0))*floor(fabs(offsetD));

    /// Accept the stub if the post-offset correction displacement is smaller than the half-window
    if ( abs(dispI - offsetI) < window )
    {
      aConfirmation = true;
      aDisplacement = 2*dispI; /// In HALF-STRIP units!
      anOffset = 2*offsetI; /// In HALF-STRIP units!
    } /// End of stub is accepted
  }
}

