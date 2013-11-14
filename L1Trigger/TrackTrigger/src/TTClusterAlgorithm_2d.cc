/*! \brief   Implementation of methods of TTClusterAlgorithm_2d
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d.h"

/// Clustering operations
template< >
void TTClusterAlgorithm_2d< Ref_PixelDigi_ >::Cluster( std::vector< std::vector< Ref_PixelDigi_ > > &output,
                                                       const std::vector< Ref_PixelDigi_ > &input ) const
{
  /// Prepare the output
  output.clear();

  /// Prepare a proper hit container
  std::map< std::pair< unsigned int, unsigned int>, pixelContainer< Ref_PixelDigi_ > >                     hitContainer;
  typename std::map< std::pair< unsigned int, unsigned int >, pixelContainer< Ref_PixelDigi_ > >::iterator centralPixel;

  /// First fill all, put the hits into a grid
  /// Loop over all hits
  typename std::vector< Ref_PixelDigi_ >::const_iterator inputIterator;
  for( inputIterator = input.begin();
       inputIterator != input.end();
       ++inputIterator )
  {
    /// Assign central Pixel
    /// Assign kill bits
    /// Assign neighbours
    hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].centrePixel = &(*inputIterator);
    hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].kill0 = false;
    hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].kill1 = false;
    hitContainer[ std::make_pair( (**inputIterator).row(), (**inputIterator).column() ) ].neighbours = 0x00;
  } /// End of loop over all hits

  /// Then search to see if neighbour hits exist
  /// Loop over all central pixels
  for( centralPixel = hitContainer.begin();
       centralPixel != hitContainer.end();
       ++centralPixel )
  {
    /// Get the coordinates
    unsigned int row = centralPixel->first.first;
    unsigned int col = centralPixel->first.second;

    /// Layout of the grid to understand what follows
    ///    a  b  c     0  1  2          -->r/phi = increasing row
    ///    d  x  e  =  3  x  4          |
    ///    f  g  h     5  6  7          V  z = decreasing column

    /// Just check if there are neighbours and, if so,
    /// assign the corresponding bit to be true/false

    /// Column +1, rows from -1 to +1
    centralPixel->second.neighbours[0] = ( hitContainer.find( std::make_pair( row-1, col+1 ) ) != hitContainer.end() );
    centralPixel->second.neighbours[1] = ( hitContainer.find( std::make_pair( row  , col+1 ) ) != hitContainer.end() );
    centralPixel->second.neighbours[2] = ( hitContainer.find( std::make_pair( row+1, col+1 ) ) != hitContainer.end() );

    /// Column 0, rows -1 and +1
    centralPixel->second.neighbours[3] = ( hitContainer.find( std::make_pair( row-1, col   ) ) != hitContainer.end() );
    centralPixel->second.neighbours[4] = ( hitContainer.find( std::make_pair( row+1, col   ) ) != hitContainer.end() );

    /// Column -1, rows from -1 to +1
    centralPixel->second.neighbours[5] = ( hitContainer.find( std::make_pair( row-1, col-1 ) ) != hitContainer.end() );
    centralPixel->second.neighbours[6] = ( hitContainer.find( std::make_pair( row  , col-1 ) ) != hitContainer.end() );
    centralPixel->second.neighbours[7] = ( hitContainer.find( std::make_pair( row+1, col-1 ) ) != hitContainer.end() );

  } /// End of loop over all central pixels

  /// Then fill the kill bits
  /// Loop over all central pixels
  for( centralPixel = hitContainer.begin();
       centralPixel != hitContainer.end();
       ++centralPixel )
  {
    /// KB 1) The first kill bit, kill0, prevents a cluster to be larger than 2 pixels in r-phi: if both columns
    /// adf and ceh contain at least one pixel over threshold each, this bit is set to 1, otherwise  it is set to 0
    /// KB 2) The second kill bit, kill1, makes the cluster to be built only if pix is in the leftmostbottom position
    /// within the cluster: if there is a pixel over threshold either in column adf or in position g,
    /// this bit is set to 1, otherwise it is set to 0

    /// Check row -1
    bool adf = centralPixel->second.neighbours[0] | centralPixel->second.neighbours[3] | centralPixel->second.neighbours[5]  ;
    /// Check row +1
    bool ceh = centralPixel->second.neighbours[2] | centralPixel->second.neighbours[4] | centralPixel->second.neighbours[7]  ;

    /// Kill bits are set here
    centralPixel->second.kill0 = ( adf & ceh );
    centralPixel->second.kill1 = ( adf | centralPixel->second.neighbours[6] );

  } /// End of loop over all central pixels

  /// Then cross check for the final kill bit
  /// Loop over all central pixels
  for( centralPixel = hitContainer.begin();
       centralPixel != hitContainer.end();
       ++centralPixel )
  {
    /// Get the coordinates
    unsigned int row = centralPixel->first.first;
    unsigned int col = centralPixel->first.second;

    /// KB 3) if at least one of the pixels, in ceh column, fired and features its kill0 = 1, let a^M      /// third kill bit kill2 be 1, otherwise set it to 0
    /// NOTE that kill2 prevents the pixel to report a cluster when looking at its size out of the 3x3
    /// pixel window under examination
    bool kill2 = false;
    typename std::map< std::pair< unsigned int, unsigned int >, pixelContainer< Ref_PixelDigi_ > >::iterator rhs;

    if ( ( rhs = hitContainer.find( std::make_pair( row+1, col-1 ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;
    if ( ( rhs = hitContainer.find( std::make_pair( row+1, col   ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;
    if ( ( rhs = hitContainer.find( std::make_pair( row+1, col+1 ) ) ) != hitContainer.end() ) kill2 |= rhs->second.kill0;

    /// If all the kill bits are fine,
    /// then the Cluster can be prepared for output
    if ( !centralPixel->second.kill0 && !centralPixel->second.kill1 && !kill2 )
    {
      /// Store the central pixel
      std::vector< Ref_PixelDigi_ > temp;
      temp.push_back( *hitContainer[ std::make_pair( row , col ) ].centrePixel );

      /// Store all the neighbours
      if( centralPixel->second.neighbours[0] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col+1 ) ].centrePixel );
      if( centralPixel->second.neighbours[1] ) temp.push_back ( *hitContainer[ std::make_pair( row  , col+1 ) ].centrePixel );
      if( centralPixel->second.neighbours[2] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col+1 ) ].centrePixel );
      if( centralPixel->second.neighbours[3] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col   ) ].centrePixel );
      if( centralPixel->second.neighbours[4] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col   ) ].centrePixel );
      if( centralPixel->second.neighbours[5] ) temp.push_back ( *hitContainer[ std::make_pair( row-1, col-1 ) ].centrePixel );
      if( centralPixel->second.neighbours[6] ) temp.push_back ( *hitContainer[ std::make_pair( row  , col-1 ) ].centrePixel );
      if( centralPixel->second.neighbours[7] ) temp.push_back ( *hitContainer[ std::make_pair( row+1, col-1 ) ].centrePixel );
      output.push_back(temp);

    } /// End of "all the kill bits are fine"
  } /// End of loop over all central pixels

  /// Eventually, if needed, do the
  /// test for double counting!
  if( mDoubleCountingTest )
  {
    std::set< std::pair< unsigned int, unsigned int > > test;
    std::set< std::pair< unsigned int, unsigned int > > doubles;
    typename std::vector< std::vector< Ref_PixelDigi_ > >::iterator outputIterator1;
    typename std::vector< Ref_PixelDigi_ >::iterator                outputIterator2;

    /// Loop over Clusters
    for ( outputIterator1 = output.begin();
          outputIterator1 != output.end();
          ++outputIterator1 )
    {
      /// Loop over Hits inside each Cluster
      for ( outputIterator2 = outputIterator1->begin();
            outputIterator2 != outputIterator1->end();
            ++outputIterator2 )
      {
        /// Are there Hits with same coordinates?
        /// If yes, put in doubles vector, else in test one
        if ( test.find( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) ) != test.end() )
          doubles.insert( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) );
        else
          test.insert( std::make_pair( (**outputIterator2).row(), (**outputIterator2).column() ) );

      } /// End of loop over Hits inside each Cluster
    } /// End of loop over Clusters

    /// If we found duplicates
    /// WARNING is it really doing something
    /// more than printout???????
    if ( doubles.size() )
    {
      std::set< std::pair< unsigned int, unsigned int> >::iterator it;
      std::stringstream errmsg;

      /// Printout double Pixel
      for ( it = doubles.begin(); it != doubles.end(); ++it )
      {
        errmsg << "Double counted pixel: (" << it->first << "," << it->second << ")\n";
      }

      /// Loop over Clusters
      for ( outputIterator1 = output.begin();
            outputIterator1 != output.end();
            ++outputIterator1 )
      {
        errmsg <<  "cluster: ";

        /// Loop over Hits inside each Cluster
        for ( outputIterator2 = outputIterator1->begin();
              outputIterator2 != outputIterator1->end();
              ++outputIterator2 )
        {
          errmsg << "| (" <<  (**outputIterator2).row() <<","<< (**outputIterator2).column()<< ") ";
        }
        errmsg << "|\n";
      } /// End of loop over Clusters

      edm::LogError("TTClusterAlgorithm_2d") << errmsg.str();

    } /// End of "if we found duplicates"
  } /// End of test for double counting
} /// End of TTClusterAlgorithm_2d< ... >::Cluster( ... )


