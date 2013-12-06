/*! \class   TTStubAlgorithm_tab2013
 *  \brief   Class for "tab2013" algorithm to be used
 *           in TTStubBuilder
 *  \details HW-friendly algorithm: layer-wise LUT.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \author Sebastien Viret
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_TAB2013_H
#define L1_TRACK_TRIGGER_STUB_ALGO_TAB2013_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <typeinfo>

template< typename T >
class TTStubAlgorithm_tab2013 : public TTStubAlgorithm< T >
{
  private:
    /// Data members
    bool        mPerformZMatchingPS;
    bool        mPerformZMatching2S;
    std::string className_;

    std::vector< double >                barrelCut;
    std::vector< std::vector< double > > ringCut;

  public:
    /// Constructor
    TTStubAlgorithm_tab2013( const StackedTrackerGeometry *aStackedTracker,
                             std::vector< double > setBarrelCut,
                             std::vector< std::vector< double > > setRingCut,
                             bool aPerformZMatchingPS, bool aPerformZMatching2S )
      : TTStubAlgorithm< T >( aStackedTracker, __func__ )
    {
      barrelCut = setBarrelCut;
      ringCut = setRingCut;
      mPerformZMatchingPS = aPerformZMatchingPS;
      mPerformZMatching2S = aPerformZMatching2S;
    }

    /// Destructor
    ~TTStubAlgorithm_tab2013(){}

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
void TTStubAlgorithm_tab2013< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                       int &aDisplacement,
                                                                       int &anOffset,
                                                                       const TTStub< Ref_PixelDigi_ > &aTTStub ) const;





/*! \class   ES_TTStubAlgorithm_tab2013
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTStubAlgorithm_tab2013 : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;

    /// Windows
    std::vector< double >                setBarrelCut;
    std::vector< std::vector< double > > setRingCut;

    /// Z-matching
    bool  mPerformZMatchingPS;
    bool  mPerformZMatching2S;

  public:
    /// Constructor
    ES_TTStubAlgorithm_tab2013( const edm::ParameterSet & p )
    {
      mPerformZMatchingPS =  p.getParameter< bool >("zMatchingPS");
      mPerformZMatching2S =  p.getParameter< bool >("zMatching2S");

      setBarrelCut = p.getParameter< std::vector< double > >("BarrelCut");

      std::vector< edm::ParameterSet > vPSet = p.getParameter< std::vector< edm::ParameterSet > >("EndcapCutSet");
      std::vector< edm::ParameterSet >::const_iterator iPSet;
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRingCut.push_back( iPSet->getParameter< std::vector< double > >("EndcapCut") );
      }
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_tab2013(){}

    /// Implement the producer
    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    { 
      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_tab2013< T >( &(*StackedTrackerGeomHandle),
                                          setBarrelCut, setRingCut,
                                          mPerformZMatchingPS, mPerformZMatching2S );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    } 

};

#endif

