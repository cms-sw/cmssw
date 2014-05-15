/*! \class   TTStubAlgorithm_a
 *  \brief   Class for "a" algorithm to be used
 *           in TTStubBuilder
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

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_A_H
#define L1_TRACK_TRIGGER_STUB_ALGO_A_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

template< typename T >
class TTStubAlgorithm_a : public TTStubAlgorithm< T >
{
  public:
    /// Constructor
    TTStubAlgorithm_a( const StackedTrackerGeometry *aStackedTracker )
      : TTStubAlgorithm< T >( aStackedTracker, __func__ ){}

    /// Destructor
    ~TTStubAlgorithm_a(){}

    /// Matching operations
    void PatternHitCorrelation( bool &aConfirmation,
                                int &aDisplacement,
                                int &anOffset,
                                const TTStub< T > &aTTStub ) const
    {
      aConfirmation = true; 
    }

}; /// Close class





/*! \class   ES_TTStubAlgorithm_a
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 18
 *
 */

template<  typename T  >
class ES_TTStubAlgorithm_a : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;

  public:
    /// Constructor
    ES_TTStubAlgorithm_a( const edm::ParameterSet & p )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_a(){}

    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_a< T >( &(*StackedTrackerGeomHandle) );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    }

};

#endif

