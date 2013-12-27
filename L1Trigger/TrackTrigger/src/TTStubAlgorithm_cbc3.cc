/*! \brief   Implementation of methods of TTStubAlgorithm_cbc3
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Ivan Reid
 *  \date   2013, Oct 16
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_cbc3.h"

/// Matching operations
template< >
void TTStubAlgorithm_cbc3< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                    int &aDisplacement, 
                                                                    int &anOffset, 
                                                                    const TTStub< Ref_PixelDigi_ > &aTTStub ) const
{
  /// Calculate average coordinates col/row for inner/outer Cluster
  /// These are already corrected for being at the center of each pixel
  MeasurementPoint mp0 = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
  MeasurementPoint mp1 = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();

  /// Get the module ID
  StackedTrackerDetId stDetId( aTTStub.getDetId() );

  //bool isPS = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->isPSModule( stDetId );

  /// Assumption: both sensors have the same pitch (as per CBC3 design...)
  const GeomDetUnit* det0 = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->idToDetUnit( stDetId, 0 );

  /// Find pixel pitch and topology related information
  const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );

  /// Get cluster position on inner sensor truncated to integer
  int myPosition = mp0.x();

  /// Get ROC information
  int chipSize = top0->rowsperroc();
  int asicNumber = myPosition / chipSize; /// ASIC in module
  int partitionSize = ceil( float(chipSize) / float( TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->getPartitionsPerRoc() ) );
  int partitionNumber = (myPosition % chipSize) / partitionSize; /// Partition in ASIC

  /// Assign the offset
  anOffset = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->getASICOffset( stDetId, asicNumber, partitionNumber );

  /// Find position and bend in HALF-STRIP units
  int aPosition = 2 * mp0.x();
  int aBend = (2 * mp1.x()) - aPosition - anOffset;

  /// Assign the displacement
  aDisplacement = (2 * mp1.x()) - aPosition;

  /// Cluster difference less predefined offset for this ASIC partition in half-strip units
  aConfirmation =  ( (abs(4 * aBend - 1)) <= (2 * TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->getDetUnitWindow(stDetId) ) );

  /// Stop here if no z-matching is required  
  if ( !mPerformZMatching2S ) // && !isPS
    return;

  //if ( !mPerformZMatchingPS && isPS )
  //  return;

  /// Check if the clusters are in the same z-segment
  const GeomDetUnit* det1 = TTStubAlgorithm< Ref_PixelDigi_ >::theStackedTracker->idToDetUnit( stDetId, 1 );
  const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
  const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );
  int cols0 = top0->ncolumns();
  int cols1 = top1->ncolumns();
  int ratio = cols0/cols1; /// This assumes the ratio is integer!
  int segment0 = floor( mp0.y() / ratio );
  if ( segment0 != floor( mp1.y() ) )
    aConfirmation = false;

}

