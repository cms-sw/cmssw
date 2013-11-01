/*! \brief   Implementation of methods of TTCluster
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \author Emmanuele Salvati
 *  \date   2013, Jul 12
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

/// Cluster width
template< >
unsigned int TTCluster<edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi > >::findWidth() const
{
  int rowMin = 99999999;
  int rowMax = 0;
  /// For broadside Clusters this is equivalent to theHits.size()
  /// but for 2d or neighbor Clusters this is only the actual size in RPhi
  for ( unsigned int i = 0; i < theHits.size(); i++ )
  {
    if ( theHits.at(i)->row() < rowMin )
      rowMin = theHits.at(i)->row();
    if ( theHits.at(i)->row() > rowMax )
      rowMax = theHits.at(i)->row();
  }
  return abs( rowMax - rowMin + 1 ); /// This takes care of 1-Pixel clusters
}

/// Get hit local coordinates
template< >
MeasurementPoint TTCluster< edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi > >::findHitLocalCoordinates( unsigned int hitIdx ) const
{
  /// NOTE in this case, DO NOT add 0.5
  /// to get the center of the pixel
  MeasurementPoint mp( theHits.at(hitIdx)->row(), theHits.at(hitIdx)->column() );
  return mp;
}

/// Unweighted average local cluster coordinates
template< >
MeasurementPoint TTCluster< edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi > >::findAverageLocalCoordinates() const
{
  double averageCol = 0.0;
  double averageRow = 0.0;

  /// Loop over the hits and calculate the average coordinates
  if ( theHits.size() != 0 )
  {
    typename std::vector< edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi > >::const_iterator hitIter;
    for ( hitIter = theHits.begin();
          hitIter != theHits.end();
          hitIter++ )
    {
      averageCol += (*hitIter)->column();
      averageRow += (*hitIter)->row();
    }
    averageCol /= theHits.size();
    averageRow /= theHits.size();
  }

  return MeasurementPoint( averageRow, averageCol );
}

