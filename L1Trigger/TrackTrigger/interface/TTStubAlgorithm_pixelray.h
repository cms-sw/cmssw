/*! \class   TTStubAlgorithm_pixelray
 *  \brief   Class for "pixelray" algorithm to be used
 *           in TTStubBuilder
 *  \details Improved backprojection w.r.t. "globalgeometry".
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Kristofer Henriksson
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_PIXELRAY_H
#define L1_TRACK_TRIGGER_STUB_ALGO_PIXELRAY_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

template < typename T >
class TTStubAlgorithm_pixelray : public TTStubAlgorithm< T >
{
  private:
    /// Data members
    double mCompatibilityScalingFactor;
    double mIPWidth;

    /// Function to get pixel ray end points
    static std::pair< double, double >* GetPixelRayEndpoints( const TTStub< T > & aTTStub,
                                                              const StackedTrackerGeometry* stackedTracker,
                                                              double scalingFactor );

  public:
    /// Constructor
    TTStubAlgorithm_pixelray( const StackedTrackerGeometry *aStackedTracker,
                              double aCompatibilityScalingFactor,
                              double aIPWidth )
      : TTStubAlgorithm< T >( aStackedTracker, __func__ )
    {
      mCompatibilityScalingFactor = aCompatibilityScalingFactor;
      mIPWidth = aIPWidth;
    }

    /// Destructor
    ~TTStubAlgorithm_pixelray(){}

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

/// Function to get pixel ray end points
template< >
std::pair< double, double >* TTStubAlgorithm_pixelray< Ref_PixelDigi_ >::GetPixelRayEndpoints( const TTStub< Ref_PixelDigi_ > & aTTStub,
                                                                                               const StackedTrackerGeometry* stackedTracker,
                                                                                               double scalingFactor );

/// Matching operations
template< >
void TTStubAlgorithm_pixelray< Ref_PixelDigi_ >::PatternHitCorrelation( bool &aConfirmation,
                                                                        int &aDisplacement,
                                                                        int &anOffset,
                                                                        const TTStub< Ref_PixelDigi_ > &aTTStub ) const;





/*! \class   ES_TTStubAlgorithm_pixelray
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Kristofer Henriksson
 *  \date   2013, Jul 18
 *
 */

template < typename T >
class  ES_TTStubAlgorithm_pixelray : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTStubAlgorithm< T > > _theAlgo;
    double                                    mPtThreshold;
    double                                    mIPWidth;

  public:
    /// Constructor
    ES_TTStubAlgorithm_pixelray( const edm::ParameterSet & p )
      : mPtThreshold( p.getParameter< double >("minPtThreshold") ),
        mIPWidth( p.getParameter< double >("ipWidth") )
    {
      setWhatProduced(this);
    }

    /// Destructor
    virtual ~ES_TTStubAlgorithm_pixelray(){}

    /// Implement the producer
    boost::shared_ptr< TTStubAlgorithm< T > > produce( const TTStubAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla( GlobalPoint(0,0,0) ).z();

      /// Calculate scaling factor based on B and Pt threshold
      double mCompatibilityScalingFactor = ( CLHEP::c_light * mMagneticFieldStrength ) / (100.0 * 2.0e+9 * mPtThreshold);

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord<StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );

      TTStubAlgorithm< T >* TTStubAlgo =
        new TTStubAlgorithm_pixelray< T >( &(*StackedTrackerGeomHandle),
                                           mCompatibilityScalingFactor,
                                           mIPWidth );

      _theAlgo = boost::shared_ptr< TTStubAlgorithm< T > >( TTStubAlgo );
      return _theAlgo;
    }

}; /// Close class

#endif

