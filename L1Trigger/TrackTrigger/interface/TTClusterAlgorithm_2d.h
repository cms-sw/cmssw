/*! \class   TTClusterAlgorithm_2d
 *  \brief   Class for "2d" algorithm to be used
 *           in TTClusterBuilder
 *  \details 2x2 clusters with some not well understood behavior
 *           in handling duplicates or larger than 2x2 ...
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_2d_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_2d_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <bitset>
#include <map>

/// Container of pixel information
template< typename T >
struct pixelContainer
{
  /// Pixel and neighbours
  const T*         centrePixel;
  std::bitset< 8 > neighbours;

  /// Kill bits (2 of many)
  bool kill0;
  bool kill1;
};

/// ..............................

template< typename T >
class TTClusterAlgorithm_2d : public TTClusterAlgorithm< T >
{
  private:
    /// Data members
    bool mDoubleCountingTest; /// This is to manage double counting

  public:
    /// Constructor
    TTClusterAlgorithm_2d( bool aDoubleCountingTest )
      : TTClusterAlgorithm< T >( __func__ )
    { 
      mDoubleCountingTest = aDoubleCountingTest;
    }

    /// Destructor
    ~TTClusterAlgorithm_2d(){}

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
template< >
void TTClusterAlgorithm_2d< Ref_Phase2TrackerDigi_ >::Cluster( std::vector< std::vector< Ref_Phase2TrackerDigi_ > > &output,
                                                       const std::vector< Ref_Phase2TrackerDigi_ > &input ) const;





/*! \class   ES_TTClusterAlgorithm_2d
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

template< typename T >
class ES_TTClusterAlgorithm_2d : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTClusterAlgorithm< T > > _theAlgo;
    bool                                         mDoubleCountingTest;

  public:
    /// Constructor
    ES_TTClusterAlgorithm_2d( const edm::ParameterSet & p )
      : mDoubleCountingTest( p.getParameter< bool >("DoubleCountingTest") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTClusterAlgorithm_2d(){}

    /// Implement the producer
    boost::shared_ptr< TTClusterAlgorithm< T > > produce( const TTClusterAlgorithmRecord & record )
    {
      TTClusterAlgorithm< T >* TTClusterAlgo =
        new TTClusterAlgorithm_2d< T >( mDoubleCountingTest );

      _theAlgo = boost::shared_ptr< TTClusterAlgorithm< T > >( TTClusterAlgo );
      return _theAlgo;
    } 

}; /// Close class

#endif

