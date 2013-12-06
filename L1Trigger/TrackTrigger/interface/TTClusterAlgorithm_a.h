/*! \class   TTClusterAlgorithm_a
 *  \brief   Class for "a" algorithm to be used
 *           in TTClusterBuilder
 *  \details Dummy: does not do anything.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_a_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_a_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

template< typename T >
class TTClusterAlgorithm_a : public TTClusterAlgorithm< T >
{
  public:
    /// Constructor
    TTClusterAlgorithm_a( const StackedTrackerGeometry *aStackedTracker )
      : TTClusterAlgorithm< T >( aStackedTracker, __func__ ){}

    /// Destructor
    ~TTClusterAlgorithm_a(){}

    /// Clustering operations  
    void Cluster( std::vector< std::vector< T > > &output,
                  const std::vector< T > &input ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Clustering operations
/// NOTE: in this case, clustering is dummy and each hit is
/// treated as a different already-ok cluster
template< typename T >
void TTClusterAlgorithm_a< T >::Cluster( std::vector< std::vector< T > > &output,
                                         const std::vector< T > &input ) const
{
  /// Prepare output
  output.clear();

  /// Loop over all hits
  typename std::vector< T >::const_iterator inputIterator;
  for( inputIterator = input.begin();
       inputIterator != input.end();
       ++inputIterator ) 
  {
    std::vector< T > temp;
    temp.push_back(*inputIterator);
    output.push_back(temp);
  } /// End of loop over all hits
} /// End of TTClusterAlgorithm_a< ... >::Cluster( ... )





/*! \class   ES_TTClusterAlgorithm_a
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

template< typename T >
class ES_TTClusterAlgorithm_a : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTClusterAlgorithm< T > > _theAlgo;

  public:
    /// Constructor
    ES_TTClusterAlgorithm_a( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTClusterAlgorithm_a(){}

    /// Implement the producer
    boost::shared_ptr< TTClusterAlgorithm< T > > produce( const TTClusterAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
 
      TTClusterAlgorithm< T >* TTClusterAlgo =
        new TTClusterAlgorithm_a< T >( &(*StackedTrackerGeomHandle) );

      _theAlgo = boost::shared_ptr< TTClusterAlgorithm< T > >( TTClusterAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

