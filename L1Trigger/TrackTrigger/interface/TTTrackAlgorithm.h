/*! \class   TTTrackAlgorithm
 *  \brief   Base class for any algorithm to be used
 *           in TTTrackBuilder
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ALGO_BASE_H
#define L1_TRACK_TRIGGER_TRACK_ALGO_BASE_H

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <sstream>
#include <map>
#include <string>
#include "classNameFinder.h"

template< typename T >
class TTTrackAlgorithm
{
  protected:
    /// Data members
    const StackedTrackerGeometry *theStackedTracker;
    std::string                  className_;

  public:
    /// Constructors
    TTTrackAlgorithm( const StackedTrackerGeometry *aStackedGeom, std::string fName )
      : theStackedTracker( aStackedGeom )
    {
      className_ = classNameFinder< T >(fName);
    }

    /// Destructor
    virtual ~TTTrackAlgorithm(){}

    /// Seed creation
    virtual void CreateSeeds( std::vector< TTTrack< T > > &output,
                              std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< T > > > > *outputSectorMap,
                              edm::Handle< std::vector< TTStub< T > > > &input ) const
    {
      output.clear();
    }

    /// Match a Stub to a Seed/Track
    virtual void AttachStubToSeed( TTTrack< T > &seed,
                                   edm::Ptr< TTStub< T > > &candidate ) const
    {
      seed.addStubPtr( candidate );
    }

    /// AM Pattern Finding
    virtual void PatternFinding() const
    {}

    /// AM Pattern Recognition
    virtual void PatternRecognition() const
    {}

    virtual unsigned int ReturnNumberOfSectors() const { return 1; } 
    virtual unsigned int ReturnNumberOfWedges() const  { return 1; }
    virtual double ReturnMagneticField() const         { return 1.0; }

    /// Fit the Track
    virtual void FitTrack( TTTrack< T > &seed ) const;

    /// Algorithm name
    virtual std::string AlgorithmName() const { return className_; }

    /// Helper methods
    double DeltaPhi( double phi1, double phi2 ) const;
    double CosineTheorem( double a, double b, double phi ) const;
    double FindRInvOver2( double rho1, double rho2, double phi1, double phi2 ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Fit the track
template< >
void TTTrackAlgorithm< Ref_PixelDigi_ >::FitTrack( TTTrack< Ref_PixelDigi_ > &seed ) const;

/// Helper methods
template< typename T >
double TTTrackAlgorithm< T >::DeltaPhi( double phi1, double phi2 ) const
{
  double deltaPhi = phi1 - phi2;
  if ( fabs(deltaPhi) >= M_PI )
  {
    if ( deltaPhi>0 )
      deltaPhi = deltaPhi - 2*M_PI;
    else
      deltaPhi = 2*M_PI + deltaPhi;
  }
  return deltaPhi;
}

template< typename T >
double TTTrackAlgorithm< T >::CosineTheorem( double a, double b, double phi ) const
{
  return sqrt( a*a + b*b - 2*a*b*cos(phi) );
}

template< typename T >
double TTTrackAlgorithm< T >::FindRInvOver2( double rho1, double rho2, double phi1, double phi2 ) const
{
  /// Calculate angle between the vectors
  double deltaPhi = this->DeltaPhi( phi1, phi2 );

  /// Apply cosine theorem to find the distance between the vectors
  double distance = this->CosineTheorem( rho1, rho2, deltaPhi );

  /// Apply sine theorem to find 1/(2R)
  return sin(deltaPhi)/distance; /// Sign is maintained to keep track of the charge
}

#endif

