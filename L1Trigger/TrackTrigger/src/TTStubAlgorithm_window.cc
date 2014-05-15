/*! \brief   Implementation of methods of TTStubAlgorithm_window
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window.h"

/// Matching operations
template< >
void TTStubAlgorithm_window< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                      int &aDisplacement, 
                                                                      int &anOffset, 
                                                                      const TTStub< Ref_PixelDigi_ > &aTTStub ) const
{
  /// Convert DetId
  StackedTrackerDetId stDetId( aTTStub.getDetId() );

  //move this out of the if to ensure that it gets set to something regardless
  aConfirmation = false;

  /// Force this to be a BARREL-only algorithm
  if ( stDetId.isEndcap() ) return;

  typename std::vector< Ref_PixelDigi_ >::const_iterator hitIter;

  /// Calculate average coordinates col/row for inner Cluster
  double averageRow = 0.0;
  double averageCol = 0.0;
  const std::vector< Ref_PixelDigi_ > &lhits0 = aTTStub.getClusterRef(0)->getHits();
  if ( lhits0.size() != 0 )
  {
    for ( hitIter = lhits0.begin();
          hitIter != lhits0.end();
          hitIter++ )
    {
      averageRow +=  (**hitIter).row();
      averageCol +=  (**hitIter).column();
    }
    averageRow /= lhits0.size();
    averageCol /= lhits0.size();
  }

  /// Calculate window based on the average row and column
  StackedTrackerWindow window = mWindowFinder->getWindow( stDetId, averageRow, averageCol );

  /// Calculate average coordinates col/row for outer Cluster
  averageRow = 0.0;
  averageCol = 0.0;
  const std::vector< Ref_PixelDigi_ > &lhits1 = aTTStub.getClusterRef(1)->getHits();
  if ( lhits1.size() != 0 )
  {
    for ( hitIter = lhits1.begin();
          hitIter != lhits1.end();
          hitIter++ )
    {
      averageRow += (**hitIter).row();
      averageCol +=  (**hitIter).column();
    }
    averageRow /= lhits1.size();
    averageCol /= lhits1.size();
  }

  /// Check if the window criteria are satisfied
  if ( ( averageRow >= window.mMinrow ) && ( averageRow <= window.mMaxrow ) &&
       ( averageCol >= window.mMincol ) && ( averageCol <= window.mMaxcol ) )
  {
    aConfirmation = true;

    /// Calculate output
    /// NOTE this assumes equal pitch in both sensors!
    MeasurementPoint mp0 = aTTStub.getClusterRef(0)->findAverageLocalCoordinates();
    MeasurementPoint mp1 = aTTStub.getClusterRef(1)->findAverageLocalCoordinates();
    aDisplacement = 2*(mp1.x() - mp0.x()); /// In HALF-STRIP units!

    /// By default, assigned as ZERO
    anOffset = 0;
  }
}

