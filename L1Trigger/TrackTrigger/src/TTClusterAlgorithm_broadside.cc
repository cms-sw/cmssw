/*! \brief   Implementation of methods of TTClusterAlgorithm_broadside
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_broadside.h"

/// Clustering operations
template< >
void TTClusterAlgorithm_broadside< Ref_PixelDigi_ >::Cluster( std::vector< std::vector< Ref_PixelDigi_ > > &output,
                                                              const std::vector< Ref_PixelDigi_ > &input ) const
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

  /// Loop over the mapped hits
  typename std::map< unsigned int, std::vector< Ref_PixelDigi_ > >::iterator mapIterator;
  mapIterator = mapHitsByColumn.begin();
  while ( mapIterator != mapHitsByColumn.end() )
  {
    /// Collect hits sharing column index and
    /// differing by 1 in row index
    typename std::vector< Ref_PixelDigi_ >::iterator inputIterator;
    inputIterator = mapIterator->second.begin();

    /// Loop over single column
    while( inputIterator != mapIterator->second.end() )
    {
      std::vector< Ref_PixelDigi_ > temp;
      temp.push_back(*inputIterator);
      inputIterator = mapIterator->second.erase(inputIterator);
      typename std::vector< Ref_PixelDigi_ >::iterator inputIterator2;
      inputIterator2 = inputIterator;

      /// Nested loop
      while( inputIterator2 != mapIterator->second.end() )
      {
        /// Check col/row and add to the cluster
        if( (temp.back()->column() == (**inputIterator2).column()) &&
            ((**inputIterator2).row() - temp.back()->row() == 1) )
        {
          temp.push_back(*inputIterator2);
          inputIterator2 = mapIterator->second.erase(inputIterator2);
        }
        else
          break;

      } /// End of nested loop

      /// Reject all clusters large than the allowed size
      if ( (mWidthCut < 1) || (int(temp.size()) <= mWidthCut) ) output.push_back(temp);
      inputIterator = inputIterator2;

    } /// End of loop over single column
    ++mapIterator;

  } /// End of loop over mapped hits
} /// End of TTClusterAlgorithm_broadside< ... >::Cluster( ... )

