/*! \brief   Implementation of methods of TTClusterAlgorithm_2d2013
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d2013.h"

/// Function to compare clusters and sort them by row
template< >
bool TTClusterAlgorithm_2d2013< Ref_PixelDigi_ >::CompareClusters( const Ref_PixelDigi_& a, const Ref_PixelDigi_& b )
{
  return ( a->row() < b->row() );
}

/// Clustering operations
template< >
void TTClusterAlgorithm_2d2013< Ref_PixelDigi_ >::Cluster( std::vector< std::vector< Ref_PixelDigi_ > > &output,
                                                           const std::vector< Ref_PixelDigi_ > &input,
                                                           bool isPS ) const
{
  /// Prepare the output
  output.clear();

  /// Prepare a proper hit container
  std::map< unsigned int, std::vector< Ref_PixelDigi_ > > mapHitsByColumn;

  /// Map all the hits by column index
  typename std::vector< Ref_PixelDigi_ >::const_iterator inputIterator;
  inputIterator = input.begin();
  while ( inputIterator != input.end() )
  {
    mapHitsByColumn[(**inputIterator).column()].push_back(*inputIterator);
    ++inputIterator;
  }

  /// 1D Clusters must be stored properly <column, first row index>
  std::map< std::pair< unsigned int, unsigned int >, std::vector< Ref_PixelDigi_ > > map1DCluByColRow;

  /// Loop over the mapped hits
  typename std::map< unsigned int, std::vector< Ref_PixelDigi_ > >::iterator mapIterHbC;
  mapIterHbC = mapHitsByColumn.begin();
  while ( mapIterHbC != mapHitsByColumn.end() )
  {
    /// Collect hits sharing column index and
    /// differing by 1 in row index
    typename std::vector< Ref_PixelDigi_ >::iterator inputIterator;
    inputIterator = mapIterHbC->second.begin();

    /// Loop over single column
    while( inputIterator != mapIterHbC->second.end() )
    {
      std::vector< Ref_PixelDigi_ > temp;
      temp.push_back(*inputIterator);
      inputIterator = mapIterHbC->second.erase(inputIterator);
      typename std::vector< Ref_PixelDigi_ >::iterator inputIterator2;
      inputIterator2 = inputIterator;

      /// Nested loop
      while( inputIterator2 != mapIterHbC->second.end() )
      {
        /// Check col/row and add to the cluster
        if( (temp.back()->column() == (**inputIterator2).column()) &&
            ((**inputIterator2).row() - temp.back()->row() == 1) )
        {
          temp.push_back(*inputIterator2);
          inputIterator2 = mapIterHbC->second.erase(inputIterator2);
        }
        else
          break;

      } /// End of nested loop

      /// Sort the vector elements by row index
      std::sort( temp.begin(), temp.end(), CompareClusters );

      /// Put the cluster in the map
      map1DCluByColRow.insert( std::make_pair( std::make_pair( mapIterHbC->first, temp.at(0)->row() ), temp ) );

      inputIterator = inputIterator2;

    } /// End of loop over single column
    ++mapIterHbC;

  } /// End of loop over mapped hits

  /// Cluster over the second dimension
  /// only in PS modules!
  typename std::map< std::pair< unsigned int, unsigned int>, std::vector< Ref_PixelDigi_ > >::iterator mapIter1DCbCR0;
  typename std::map< std::pair< unsigned int, unsigned int>, std::vector< Ref_PixelDigi_ > >::iterator mapIter1DCbCR1;
  mapIter1DCbCR0 = map1DCluByColRow.begin();
  unsigned int lastCol = mapIter1DCbCR0->first.first;

  while ( mapIter1DCbCR0 != map1DCluByColRow.end() )
  {
    /// Add the hits
    std::vector< Ref_PixelDigi_ > candCluster;
    candCluster.insert( candCluster.end(), mapIter1DCbCR0->second.begin(), mapIter1DCbCR0->second.end() );

    if ( isPS )
    {
      /// Loop over the other elements of the map
      mapIter1DCbCR1 = map1DCluByColRow.begin();

      while ( mapIter1DCbCR1 != map1DCluByColRow.end() )
      {
        /// Skip same element
        if ( mapIter1DCbCR1 == mapIter1DCbCR0 )
        {
          ++mapIter1DCbCR1;
          continue;
        }

        /// Skip non-contiguous column
        if ( fabs( mapIter1DCbCR1->first.first - lastCol ) != 1 )
        {
          ++mapIter1DCbCR1;
          continue;
        }

        /// Column is contiguous
        /// Update the "last column index"
        /// This should be safe as maps are sorted structures by construction
        lastCol = mapIter1DCbCR1->first.first;

        /// Check that the cluster is good to be clustered
        /// Get first row
        unsigned int iRow0 = mapIter1DCbCR0->first.second;
        unsigned int iRow1 = mapIter1DCbCR1->first.second;

        /// Get the max row in the cluster
        unsigned int jRow0 = mapIter1DCbCR0->second.back()->row();
        unsigned int jRow1 = mapIter1DCbCR1->second.back()->row();

        /// Check if they overlap
        if ( ( iRow1 >= iRow0 && iRow1 <= jRow0 ) ||
             ( jRow1 >= iRow0 && jRow1 <= jRow0 ) )
        {
          /// If so, add the hits to the cluster!
          candCluster.insert( candCluster.end(), mapIter1DCbCR1->second.begin(), mapIter1DCbCR1->second.end() );
          map1DCluByColRow.erase( mapIter1DCbCR1++ );
        }
        else
        {
          ++mapIter1DCbCR1;
        }
      } /// End of nested loop

      map1DCluByColRow.erase( mapIter1DCbCR0++ );

      /// Check output
      /// Sort the vector by row index
      std::sort( candCluster.begin(), candCluster.end(), CompareClusters );

      if ( fabs( candCluster.at(0)->row() - candCluster.back()->row() ) < mWidthCut || /// one should add 1 to use <=
           mWidthCut < 1 )
      {
        output.push_back( candCluster );
      }
    } /// End of isPS
    else
    {
      map1DCluByColRow.erase( mapIter1DCbCR0++ );

      /// Check output
      /// Sort the vector by row index
      std::sort( candCluster.begin(), candCluster.end(), CompareClusters );

      if ( fabs( candCluster.at(0)->row() - candCluster.back()->row() ) < mWidthCut || /// one should add 1 to use <=
           mWidthCut < 1 )
      {
        output.push_back( candCluster );
      }
    } /// End of non-PS case
  } /// End of loop over mapped 1D Clusters
}

