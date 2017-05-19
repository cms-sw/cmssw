/*! \brief   Implementation of methods of TTClusterAlgorithm_neighbor
 *  \details Here, in the source file, the methods which do depend
 *           on the specific type <T> that can fit the template.
 *
 *  \author Kristofer Henriksson
 *  \date   2013, Jul 15
 *
 */

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_neighbor.h"

/// Clustering operations
/// Specialize template for Phase2TrackerDigis
template< >
void TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::Cluster( std::vector< std::vector< Ref_Phase2TrackerDigi_ > > &output,
                                                             const std::vector< Ref_Phase2TrackerDigi_ > &input ) const
{
  /// Prepare output
  output.clear();

  /// Loop over all input hits and delete
  /// them once clustered
  std::vector< bool > used( input.size(), false );

  for ( unsigned int i = 0; i < input.size(); i++ )
  {
    if ( used[i] )
      continue;

    std::vector< Ref_Phase2TrackerDigi_ > cluster;
    cluster.push_back(input[i]);
    used[i] = true;
    if ( i < input.size()-1 )
    {
      addNeighbors( cluster, input, i+1, used );
    }
    output.push_back( cluster );
  } /// End of iteration
} /// End of Clustering Operations

/// Check if the hit is a neighbour
/// Specialize template for Phase2TrackerDigis
template< >
bool TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::isANeighbor( const Ref_Phase2TrackerDigi_& center,
                                                                 const Ref_Phase2TrackerDigi_& mayNeigh ) const
{
  unsigned int rowdist = std::abs((int)(center->row()) - (int)(mayNeigh->row()));
  unsigned int coldist = std::abs((int)(center->column()) - (int)(mayNeigh->column()));
  return rowdist <= 1 && coldist <= 1;
}

/// Add neighbours to the cluster
/// Specialize template for Phase2TrackerDigis
template< >
void TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::addNeighbors( std::vector< Ref_Phase2TrackerDigi_ >& cluster,
                                                                  const std::vector< Ref_Phase2TrackerDigi_ >& input,
                                                                  unsigned int startVal,
                                                                  std::vector< bool >& used) const
{
  /// This following line is necessary to ensure the
  /// iterators afterward remain valid.
  cluster.reserve( input.size() );
  typename std::vector< Ref_Phase2TrackerDigi_ >::iterator clusIter;
  typename std::vector< Ref_Phase2TrackerDigi_ >::iterator inIter;

  /// Loop over hits
  for ( clusIter = cluster.begin();
        clusIter < cluster.end();
        clusIter++ )
  {
    /// Loop over candidate neighbours
    for ( unsigned int i=startVal; i<input.size(); i++)
    {
      /// Is it really a neighbour?
      if ( isANeighbor(*clusIter, input[i]) )
      {
        cluster.push_back(input[i]);
        used[i]=true;
      }
    } /// End of loop over candidate neighbours
  } /// End of loop over hits
}

