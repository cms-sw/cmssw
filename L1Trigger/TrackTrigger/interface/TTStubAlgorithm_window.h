/*! \class   TTStubAlgorithm_window
 *  \brief   Class for "window" algorithm to be used
 *           in TTStubBuilder
 *  \details Makes mixed use of local and global coordinates to accept the stub
 *           above threshold and to backproject it to the luminous region.
 *           Local coordinates are used to open a window, but the procedure is
 *           somehow "tricky"...
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_WINDOW_H
#define L1_TRACK_TRIGGER_STUB_ALGO_WINDOW_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window_WindowFinder.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

template< typename T >
class TTStubAlgorithm_window : public TTStubAlgorithm< T >
{
  private:
    /// Data members
    WindowFinder    *mWindowFinder;

  public:
    /// Constructor
    TTStubAlgorithm_window( const StackedTrackerGeometry *aStackedTracker,
                            double aPtScalingFactor,
                            double aIPwidth,
                            double aRowResolution,
                            double aColResolution )
      : TTStubAlgorithm< T >( aStackedTracker, __func__ ),
        mWindowFinder( new WindowFinder( aStackedTracker,
                                         aPtScalingFactor,
                                         aIPwidth,
                                         aRowResolution,
                                         aColResolution ) ){}

    /// Destructor
    ~TTStubAlgorithm_window(){}

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
void TTStubAlgorithm_window< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                      int &aDisplacement,
                                                                      int &anOffset,
                                                                      const TTStub< Ref_PixelDigi_ > &aTTStub ) const;





/*! \class   ES_TTStubAlgorithm_window
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTStubAlgorithm_window : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;
    double mPtThreshold;
    double mIPWidth;
    double mRowResolution;
    double mColResolution;

  public:
    /// Constructor
    ES_TTStubAlgorithm_window( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") ),
        mIPWidth( p.getParameter< double >("ipWidth") ),
        mRowResolution( p.getParameter< double >("RowResolution") ),
        mColResolution( p.getParameter< double >("ColResolution") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_window(){}

    /// Implement the producer
    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    { 
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla( GlobalPoint(0,0,0) ).z();

      /// Calculate scaling factor based on B and Pt threshold
      double mPtScalingFactor = 0.0015 * mMagneticFieldStrength / mPtThreshold;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  
      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_window< T >( &(*StackedTrackerGeomHandle),
                                         mPtScalingFactor,
                                         mIPWidth,
                                         mRowResolution,
                                         mColResolution );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    } 

};

#endif

