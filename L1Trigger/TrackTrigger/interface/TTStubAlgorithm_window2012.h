/*! \class   TTStubAlgorithm_window2012
 *  \brief   Class for "window2012" algorithm to be used
 *           in TTStubBuilder
 *  \details Makes use of local coordinates and module global position to accept the stub.
 *           First version written in 2012 with approximations.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_WINDOW2012_H
#define L1_TRACK_TRIGGER_STUB_ALGO_WINDOW2012_H

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
#include "CLHEP/Units/PhysicalConstants.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <typeinfo>

template< typename T >
class TTStubAlgorithm_window2012 : public TTStubAlgorithm< T >
{
  private:
    /// Data members
    double      mPtScalingFactor;
    std::string className_;

  public:
    /// Constructor
    TTStubAlgorithm_window2012( const StackedTrackerGeometry *aStackedTracker,
                                double aPtScalingFactor )
      : TTStubAlgorithm< T >( aStackedTracker,__func__ )
    {
      mPtScalingFactor = aPtScalingFactor;
    }

    /// Destructor
    ~TTStubAlgorithm_window2012(){}

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
void TTStubAlgorithm_window2012< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                          int &aDisplacement, 
                                                                          int &anOffset, 
                                                                          const TTStub< Ref_PixelDigi_ > &aTTStub ) const;




/*! \class   ES_TTStubAlgorithm_window2012
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTStubAlgorithm_window2012 : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;

  public:
    /// Constructor
    ES_TTStubAlgorithm_window2012( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") )
    //                                mIPWidth( p.getParameter<double>("ipWidth") ),
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_window2012(){}

    /// Implement the producer
    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    { 
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

      /// Calculate scaling factor based on B and Pt threshold
      //double mPtScalingFactor = 0.0015*mMagneticFieldStrength/mPtThreshold;
      //double mPtScalingFactor = (CLHEP::c_light * mMagneticFieldStrength) / (100.0 * 2.0e+9 * mPtThreshold);
      double mPtScalingFactor = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0*0.0015/mPtThreshold;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_window2012< T >( &(*StackedTrackerGeomHandle),
                                                               mPtScalingFactor );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    } 

};

#endif

