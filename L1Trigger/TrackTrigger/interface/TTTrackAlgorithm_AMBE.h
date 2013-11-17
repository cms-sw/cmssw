/*! \class   TTTrackAlgorithm_AMBE
 *  \brief   Skeleton for AM-based track finder simulation.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \author Sebastien Viret
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ALGO_ASSOBE_H
#define L1_TRACK_TRIGGER_TRACK_ALGO_ASSOBE_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>

template< typename T >
class TTTrackAlgorithm_AMBE : public TTTrackAlgorithm< T >
{
  private :
    /// Data members
    double       mMagneticField;
    unsigned int nSectors;
    unsigned int nWedges;

  public:
    /// Constructors
    TTTrackAlgorithm_AMBE( const StackedTrackerGeometry *aStackedGeom,
                           double aMagneticField,
                           unsigned int aSectors,
                           unsigned int aWedges )
      : TTTrackAlgorithm< T > ( aStackedGeom, __func__ )
    {
      mMagneticField = aMagneticField;
      nSectors = aSectors;
      nWedges = aWedges;
    }

    /// Destructor
    ~TTTrackAlgorithm_AMBE(){}

    /// Pattern Finding
    void PatternFinding() const;

    /// Pattern Recognition
    void PatternRecognition() const;

    /// Return the number of Sectors
    unsigned int ReturnNumberOfSectors() const { return nSectors; } /// Phi
    unsigned int ReturnNumberOfWedges() const  { return nWedges; } /// Eta

    /// Return the value of the magnetic field
    double ReturnMagneticField() const { return mMagneticField; }

    /// Fit the Track
    void FitTrack( TTTrack< T > &track ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

template< typename T >
void TTTrackAlgorithm_AMBE< T >::PatternFinding() const
{
  std::cerr << "Pattern Finding" << std::endl;
}

template< typename T >
void TTTrackAlgorithm_AMBE< T >::PatternRecognition() const
{
  std::cerr << "Pattern Recognition" << std::endl;
}

/// Fit the track
template< typename T >
void TTTrackAlgorithm_AMBE< T >::FitTrack( TTTrack< T > &track ) const
{
  std::cerr << "HOUGH!!!" << std::endl;
}





/*! \class   ES_TTTrackAlgorithm_AMBE
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTTrackAlgorithm_AMBE : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTTrackAlgorithm< T > > _theAlgo;

    /// Number of Sectors
    unsigned int  mSectors;
    unsigned int  mWedges;

  public:
    /// Constructor
    ES_TTTrackAlgorithm_AMBE( const edm::ParameterSet & p )
      : mSectors( p.getParameter< int >("NumSectors") ),
        mWedges( p.getParameter< int >("NumWedges") )
    {
      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTTrackAlgorithm_AMBE() {}

    /// Implement the producer
    boost::shared_ptr< TTTrackAlgorithm< T > > produce( const TTTrackAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla( GlobalPoint(0,0,0) ).z();
      double mMagneticFieldRounded = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TTTrackAlgorithm< T >* TTTrackAlgo =
        new TTTrackAlgorithm_AMBE< T >( &(*StackedTrackerGeomHandle),
                                        mMagneticFieldRounded,
                                        mSectors,
                                        mWedges );

      _theAlgo = boost::shared_ptr< TTTrackAlgorithm< T > >( TTTrackAlgo );
      return _theAlgo;
    }

};

#endif

