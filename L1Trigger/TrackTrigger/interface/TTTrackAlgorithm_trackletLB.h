/*! \class   TTTrackAlgorithm_trackletLB
 *  \brief   Tracklet-based algorithm for the LB layout.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Anders Ryd
 *  \author Emmanuele Salvati
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ALGO_EXACTLB_H
#define L1_TRACK_TRIGGER_TRACK_ALGO_EXACTLB_H

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
class TTTrackAlgorithm_trackletLB : public TTTrackAlgorithm< T >
{
  private :
    /// Data members
    double       mMagneticField;
    unsigned int nSectors;
    unsigned int nWedges;

    std::vector< std::vector< double > > tableRPhi;
    std::vector< std::vector< double > > tableZ;

  public:
    /// Constructors
    TTTrackAlgorithm_trackletLB( const StackedTrackerGeometry *aStackedGeom,
                                 double aMagneticField,
                                 unsigned int aSectors,
                                 unsigned int aWedges,
                                 std::vector< std::vector< double > > aTableRPhi,
                                 std::vector< std::vector< double > > aTableZ )
      : TTTrackAlgorithm< T > ( aStackedGeom, __func__ )
    {
      mMagneticField = aMagneticField;
      nSectors = aSectors;
      nWedges = aWedges;

      tableRPhi = aTableRPhi;
      tableZ = aTableZ;
    }

    /// Destructor
    ~TTTrackAlgorithm_trackletLB(){}

    /// Seed creation
    void CreateSeeds( std::vector< TTTrack< T > > &output,
                      std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< T > > > > *outputSectorMap,
                      edm::Handle< std::vector< TTStub< T > > > &input ) const;

    /// Match a Stub to a Seed/Track
    void AttachStubToSeed( TTTrack< T > &seed,
                           edm::Ptr< TTStub< T > > &candidate ) const;

    /// Return the number of Sectors
    unsigned int ReturnNumberOfSectors() const { return nSectors; } /// Phi
    unsigned int ReturnNumberOfWedges()  const { return nWedges; } /// Eta

    /// Return the value of the magnetic field
    double ReturnMagneticField() const { return mMagneticField; }

    /// Fit the Track
    /// Take it from the parent class

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Create Seeds
template< >
void TTTrackAlgorithm_trackletLB< Ref_PixelDigi_ >::CreateSeeds( std::vector< TTTrack< Ref_PixelDigi_ > > &output,
                                                                 std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *outputSectorMap,
                                                                 edm::Handle< std::vector< TTStub< Ref_PixelDigi_ > > > &input ) const;

/// Match a Stub to a Seed/Track
template< >
void TTTrackAlgorithm_trackletLB< Ref_PixelDigi_ >::AttachStubToSeed( TTTrack< Ref_PixelDigi_ > &seed, edm::Ptr< TTStub< Ref_PixelDigi_ > > &candidate ) const;






/*! \class   ES_TTTrackAlgorithm_trackletLB
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTTrackAlgorithm_trackletLB : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTTrackAlgorithm< T > > _theAlgo;

    /// Number of Sectors
    unsigned int  mSectors;
    unsigned int  mWedges;

    /// projection windows
    std::vector< std::vector< double > > setRhoPhiWin;
    std::vector< std::vector< double > > setZWin;

  public:
    /// Constructor
    ES_TTTrackAlgorithm_trackletLB( const edm::ParameterSet & p )
      : mSectors( p.getParameter< int >("NumSectors") ), mWedges( p.getParameter< int >("NumWedges") )
    {
      std::vector< edm::ParameterSet > vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindows");
      std::vector< edm::ParameterSet >::const_iterator iPSet;
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWin.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWin.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
      }

      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTTrackAlgorithm_trackletLB() {}

    /// Implement the producer
    boost::shared_ptr< TTTrackAlgorithm< T > > produce( const TTTrackAlgorithmRecord & record )
    {
      /// Get magnetic field
      edm::ESHandle< MagneticField > magnet;
      record.getRecord< IdealMagneticFieldRecord >().get(magnet);
      double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();
      double mMagneticFieldRounded = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;

      edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
      record.getRecord< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );

      TTTrackAlgorithm< T >* TTTrackAlgo =
        new TTTrackAlgorithm_trackletLB< T >( &(*StackedTrackerGeomHandle),
                                              mMagneticFieldRounded,
                                              mSectors,
                                              mWedges,
                                              setRhoPhiWin,
                                              setZWin );

      _theAlgo = boost::shared_ptr< TTTrackAlgorithm< T > >( TTTrackAlgo );
      return _theAlgo;
    }

};

#endif

