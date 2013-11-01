/*! \class   TTTrackAlgorithm_trackletBE
 *  \brief   Tracklet-based algorithm for the BE layout.
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

#ifndef L1_TRACK_TRIGGER_TRACK_ALGO_TRACKLETBE_H
#define L1_TRACK_TRIGGER_TRACK_ALGO_TRACKLETBE_H

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
class TTTrackAlgorithm_trackletBE : public TTTrackAlgorithm< T >
{
  private :
    /// Data members
    double       mMagneticField;
    unsigned int nSectors;
    unsigned int nWedges;

    std::vector< std::vector< double > > tableRPhiBB;
    std::vector< std::vector< double > > tableZBB;
    std::vector< std::vector< double > > tableRPhiBE;
    std::vector< std::vector< double > > tableZBE;
    std::vector< std::vector< double > > tableRPhiEB;
    std::vector< std::vector< double > > tableZEB;
    std::vector< std::vector< double > > tableRPhiEE;
    std::vector< std::vector< double > > tableZEE;

    std::vector< std::vector< double > > tableRPhiBE_PS;
    std::vector< std::vector< double > > tableZBE_PS;
    std::vector< std::vector< double > > tableRPhiEB_PS;
    std::vector< std::vector< double > > tableZEB_PS;
    std::vector< std::vector< double > > tableRPhiEE_PS;
    std::vector< std::vector< double > > tableZEE_PS;

  public:
    /// Constructors
    TTTrackAlgorithm_trackletBE( const StackedTrackerGeometry *aStackedGeom,
                                 double aMagneticField,
                                 unsigned int aSectors,
                                 unsigned int aWedges,
                                 std::vector< std::vector< double > > aTableRPhiBB,
                                 std::vector< std::vector< double > > aTableZBB,
                                 std::vector< std::vector< double > > aTableRPhiBE,
                                 std::vector< std::vector< double > > aTableZBE,
                                 std::vector< std::vector< double > > aTableRPhiBE_PS, 
                                 std::vector< std::vector< double > > aTableZBE_PS, 
                                 std::vector< std::vector< double > > aTableRPhiEB,
                                 std::vector< std::vector< double > > aTableZEB,
                                 std::vector< std::vector< double > > aTableRPhiEB_PS,
                                 std::vector< std::vector< double > > aTableZEB_PS,
                                 std::vector< std::vector< double > > aTableRPhiEE,
                                 std::vector< std::vector< double > > aTableZEE,
                                 std::vector< std::vector< double > > aTableRPhiEE_PS,
                                 std::vector< std::vector< double > > aTableZEE_PS )
      : TTTrackAlgorithm< T > ( aStackedGeom, __func__ )
    {
      mMagneticField = aMagneticField;
      nSectors = aSectors;
      nWedges = aWedges;

      tableRPhiBB = aTableRPhiBB;
      tableZBB = aTableZBB;
      tableRPhiBE = aTableRPhiBE;
      tableZBE = aTableZBE;
      tableRPhiEB = aTableRPhiEB;
      tableZEB = aTableZEB;
      tableRPhiEE = aTableRPhiEE;
      tableZEE = aTableZEE;

      tableRPhiBE_PS = aTableRPhiBE_PS;
      tableZBE_PS = aTableZBE_PS;
      tableRPhiEB_PS = aTableRPhiEB_PS;
      tableZEB_PS = aTableZEB_PS;
      tableRPhiEE_PS = aTableRPhiEE_PS;
      tableZEE_PS = aTableZEE_PS;
    }

    /// Destructor
    ~TTTrackAlgorithm_trackletBE(){}

    /// Seed creation
    void CreateSeeds( std::vector< TTTrack< T > > &output,
                      std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< T > > > > *outputSectorMap,
                      edm::Handle< std::vector< TTStub< T > > > &input ) const;

    /// Match a Stub to a Seed/Track
    void AttachStubToSeed( TTTrack< T > &seed,
                           edm::Ptr< TTStub< T > > &candidate ) const;

    /// Return the number of Sectors
    unsigned int ReturnNumberOfSectors() const { return nSectors; } /// Phi
    unsigned int ReturnNumberOfWedges() const  { return nWedges; } /// Eta

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
void TTTrackAlgorithm_trackletBE< Ref_PixelDigi_ >::CreateSeeds( std::vector< TTTrack< Ref_PixelDigi_ > > &output,
                                                                 std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > *outputSectorMap,
                                                                 edm::Handle< std::vector< TTStub< Ref_PixelDigi_ > > > &input ) const;

/// Match a Stub to a Seed/Track
template< >
void TTTrackAlgorithm_trackletBE< Ref_PixelDigi_ >::AttachStubToSeed( TTTrack< Ref_PixelDigi_ > &seed,
                                                                      edm::Ptr< TTStub< Ref_PixelDigi_ > > &candidate ) const;





/*! \class   ES_TTTrackAlgorithm_trackletBE
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template< typename T >
class ES_TTTrackAlgorithm_trackletBE : public edm::ESProducer
{
  private:
    /// Data members
    boost::shared_ptr< TTTrackAlgorithm< T > > _theAlgo;

    /// Number of Sectors
    unsigned int mSectors;
    unsigned int mWedges;

    /// Projection windows
    std::vector< std::vector< double > > setRhoPhiWinBB;
    std::vector< std::vector< double > > setZWinBB;
    std::vector< std::vector< double > > setRhoPhiWinBE;
    std::vector< std::vector< double > > setZWinBE;
    std::vector< std::vector< double > > setRhoPhiWinEB;
    std::vector< std::vector< double > > setZWinEB;
    std::vector< std::vector< double > > setRhoPhiWinEE;
    std::vector< std::vector< double > > setZWinEE;

    /// PS Modules variants
    /// NOTE these are not needed for the Barrel-Barrel case
    std::vector< std::vector< double > > setRhoPhiWinBE_PS;
    std::vector< std::vector< double > > setZWinBE_PS;
    std::vector< std::vector< double > > setRhoPhiWinEB_PS;
    std::vector< std::vector< double > > setZWinEB_PS;
    std::vector< std::vector< double > > setRhoPhiWinEE_PS;
    std::vector< std::vector< double > > setZWinEE_PS;

  public:
    /// Constructor
    ES_TTTrackAlgorithm_trackletBE( const edm::ParameterSet & p )
      : mSectors( p.getParameter< int >("NumSectors") ),
        mWedges( p.getParameter< int >("NumWedges") )
    {
      std::vector< edm::ParameterSet > vPSet;
      std::vector< edm::ParameterSet >::const_iterator iPSet;

      vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindowsBarrelBarrel");
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWinBB.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWinBB.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
      }

      vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindowsBarrelEndcap");
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWinBE.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWinBE.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
        setRhoPhiWinBE_PS.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWinPS") );
        setZWinBE_PS.push_back( iPSet->getParameter< std::vector< double > >("ZWinPS") );
      }

      vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindowsEndcapBarrel");
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWinEB.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWinEB.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
        setRhoPhiWinEB_PS.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWinPS") );
        setZWinEB_PS.push_back( iPSet->getParameter< std::vector< double > >("ZWinPS") );
      }

      vPSet = p.getParameter< std::vector< edm::ParameterSet > >("ProjectionWindowsEndcapEndcap");
      for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
      {
        setRhoPhiWinEE.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWin") );
        setZWinEE.push_back( iPSet->getParameter< std::vector< double > >("ZWin") );
        setRhoPhiWinEE_PS.push_back( iPSet->getParameter< std::vector< double > >("RhoPhiWinPS") );
        setZWinEE_PS.push_back( iPSet->getParameter< std::vector< double > >("ZWinPS") );
      }

      setWhatProduced( this );
    }

    /// Destructor
    virtual ~ES_TTTrackAlgorithm_trackletBE() {}

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
        new TTTrackAlgorithm_trackletBE< T >( &(*StackedTrackerGeomHandle),
                                              mMagneticFieldRounded,
                                              mSectors,
                                              mWedges,
                                              setRhoPhiWinBB,
                                              setZWinBB,
                                              setRhoPhiWinBE,
                                              setZWinBE,
                                              setRhoPhiWinBE_PS,
                                              setZWinBE_PS,
                                              setRhoPhiWinEB,
                                              setZWinEB,
                                              setRhoPhiWinEB_PS,
                                              setZWinEB_PS,
                                              setRhoPhiWinEE,
                                              setZWinEE,
                                              setRhoPhiWinEE_PS,
                                              setZWinEE_PS );

      _theAlgo = boost::shared_ptr< TTTrackAlgorithm< T > >( TTTrackAlgo );
      return _theAlgo;
    }

};

#endif

