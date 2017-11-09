/*! \class   TTClusterAlgorithm_neighbor
 *  \brief   Class for "neighbor" algorithm to be used
 *           in TTClusterBuilder
 *  \details This is a greedy clustering to be
 *           used for diagnostic purposes, which
 *           will make clusters as large as
 *           possible by including all contiguous
 *           hits in a single cluster.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Kristofer Henriksson
 *  \date   2013, Jul 15
 *
 */

#ifndef CLUSTERING_ALGORITHM_NEIGHBOR_H
#define CLUSTERING_ALGORITHM_NEIGHBOR_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmRecord.h"

#include <string>
#include <cstdlib>
#include <map>

template< typename T >
class TTClusterAlgorithm_neighbor : public TTClusterAlgorithm< T >
{
  private:
    /// Data members
    /// Other stuff
    
  public:
    /// Constructor
    TTClusterAlgorithm_neighbor( )
      : TTClusterAlgorithm< T >( __func__ ){}

    /// Destructor
    ~TTClusterAlgorithm_neighbor() override{}

    /// Clustering operations  
    void Cluster( std::vector< std::vector< T > > &output,
                  const std::vector< T > &input) const override;

    /// Needed for neighbours
    bool isANeighbor( const T& center, const T& mayNeigh) const;
    void addNeighbors( std::vector< T >& cluster, const std::vector< T >& input, unsigned int start, std::vector<bool> &masked ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Clustering operations
template< >
void TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::Cluster( std::vector<std::vector< Ref_Phase2TrackerDigi_ > > &output,
                                                             const std::vector< Ref_Phase2TrackerDigi_ > &input ) const;

/// Check if the hit is a neighbour
template< >
bool TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::isANeighbor( const Ref_Phase2TrackerDigi_& center,
                                                                 const Ref_Phase2TrackerDigi_& mayNeigh ) const;

/// Add neighbours to the cluster
template< >
void TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ >::addNeighbors( std::vector< Ref_Phase2TrackerDigi_ >& cluster,
                                                                  const std::vector< Ref_Phase2TrackerDigi_ >& input,
				                                  unsigned int startVal,
                                                                  std::vector< bool >& used) const;






/*! \class   ES_TTClusterAlgorithm_neighbor
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Kristofer Henriksson
 *  \date   2013, Jul 15
 *
 */

template< typename T >
class ES_TTClusterAlgorithm_neighbor : public edm::ESProducer
{
  private:
    /// Data members
    std::shared_ptr< TTClusterAlgorithm< T > > _theAlgo;    

  public:
    /// Constructor
    ES_TTClusterAlgorithm_neighbor( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    ~ES_TTClusterAlgorithm_neighbor() override{}

    /// Implement the producer
    std::shared_ptr< TTClusterAlgorithm< T > > produce( const TTClusterAlgorithmRecord & record )
    { 
      TTClusterAlgorithm< T >* TTClusterAlgo =
        new TTClusterAlgorithm_neighbor< T >( );

      _theAlgo = std::shared_ptr< TTClusterAlgorithm< T > >( TTClusterAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

