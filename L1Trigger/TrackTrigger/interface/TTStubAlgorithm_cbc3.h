/*! \class   TTStubAlgorithm_cbc3
 *  \brief   Class for "cbc3" algorithm to be used
 *           in TTStubBuilder
 *  \details HW emulation.
 *
 *  \author Ivan Reid
 *  \date   2013, Oct 16
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_CBC3_H
#define L1_TRACK_TRIGGER_STUB_ALGO_CBC3_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <typeinfo>

template< typename T >
class TTStubAlgorithm_cbc3 : public TTStubAlgorithm< T >
{
  private:
    /// Data members
    //bool        mPerformZMatchingPS;
    bool        mPerformZMatching2S;
    std::string className_;

  public:
    /// Constructor
    TTStubAlgorithm_cbc3( const StackedTrackerGeometry *aStackedTracker,
                          //bool aPerformZMatchingPS,
                          bool aPerformZMatching2S )
      : TTStubAlgorithm< T >( aStackedTracker, __func__ )
    {
      //mPerformZMatchingPS = aPerformZMatchingPS;
      mPerformZMatching2S = aPerformZMatching2S;
    }

    /// Destructor
    ~TTStubAlgorithm_cbc3(){}

    /// Matching operations
    void PatternHitCorrelation( bool &aConfirmation,
                                int &aDisplacement,
                                int &anOffset,
                                const TTStub< T > &aTTStub ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Matching operations
template< >
void TTStubAlgorithm_cbc3< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                    int &aDisplacement,
                                                                    int &anOffset,
                                                                    const TTStub< Ref_PixelDigi_ > &aTTStub ) const;





/*! \class   ES_TTStubAlgorithm_cbc3
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTStubAlgorithm_cbc3 : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;

    /// Z-matching
    //bool  mPerformZMatchingPS;
    bool  mPerformZMatching2S;

  public:
    /// Constructor
    ES_TTStubAlgorithm_cbc3( const edm::ParameterSet & p )
    {
      //mPerformZMatchingPS =  p.getParameter< bool >("zMatchingPS");
      mPerformZMatching2S =  p.getParameter< bool >("zMatching2S");

      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_cbc3(){}

    /// Implement the producer
    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_cbc3< T >( &(*StackedTrackerGeomHandle),
                                       //mPerformZMatchingPS,
                                       mPerformZMatching2S );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    } 

};

#endif

