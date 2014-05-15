/*! \class   TTClusterAlgorithm_broadside
 *  \brief   Class for "broadside" algorithm to be used
 *           in TTClusterBuilder
 *  \details Makes 1D clusters in rphi as wide as you want.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_BROADSIDE_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_BROADSIDE_H

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
class TTClusterAlgorithm_broadside : public TTClusterAlgorithm< T >
{
  private:
    /// Data members
    int mWidthCut; /// Cluster max width

  public:
    /// Constructor
    TTClusterAlgorithm_broadside( const StackedTrackerGeometry *aStackedTracker, int aWidthCut )
      : TTClusterAlgorithm< T >( aStackedTracker, __func__ )
    {
      mWidthCut = aWidthCut;
    }

    /// Destructor
    ~TTClusterAlgorithm_broadside(){}

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
void TTClusterAlgorithm_broadside< Ref_PixelDigi_ >::Cluster( std::vector< std::vector< Ref_PixelDigi_ > > &output,
                                                              const std::vector< Ref_PixelDigi_ > &input ) const;





/*! \class   ES_TTClusterAlgorithm_broadside
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

template< typename T >
class  ES_TTClusterAlgorithm_broadside: public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTClusterAlgorithm< T > > _theAlgo;
    int                                          mWidthCut;

  public:
    /// Constructor
    ES_TTClusterAlgorithm_broadside( const edm::ParameterSet & p )
      : mWidthCut( p.getParameter< int >("WidthCut") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTClusterAlgorithm_broadside(){}

    /// Implement the producer
    boost::shared_ptr< TTClusterAlgorithm< T > > produce( const TTClusterAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      TTClusterAlgorithm< T >* TTClusterAlgo =
        new TTClusterAlgorithm_broadside< T >( &(*StackedTrackerGeomHandle), mWidthCut );

      _theAlgo = boost::shared_ptr< TTClusterAlgorithm< T > >( TTClusterAlgo );
      return _theAlgo;
    } 

}; /// Close class

#endif

