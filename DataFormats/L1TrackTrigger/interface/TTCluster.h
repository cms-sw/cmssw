/*! \class   TTCluster
 *  \brief   Class to store the L1 Track Trigger clusters
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \author Emmanuele Salvati
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_FORMAT_H
#define L1_TRACK_TRIGGER_CLUSTER_FORMAT_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" /// NOTE: this is needed even if it seems not

template< typename T >
class TTCluster
{
  public:
    /// Constructors
    TTCluster();
    TTCluster( std::vector< T > aHits,
               DetId aDetId,
               unsigned int aStackMember,
               bool storeLocal );

    /// Destructor
    ~TTCluster();

    /// Data members:   getABC( ... )
    /// Helper methods: findABC( ... )

    /// Hits in the Cluster
    std::vector< T > getHits() const                   { return theHits; }
    void             setHits( std::vector< T > aHits ) { theHits = aHits; }

    /// Detector element
    DetId        getDetId() const         { return theDetId; }
    void         setDetId( DetId aDetId ) { theDetId = aDetId; }
    unsigned int getStackMember() const                      { return theStackMember; }
    void         setStackMember( unsigned int aStackMember ) { theStackMember = aStackMember; }

    /// Rows and columns to get rid of Digi collection
    std::vector< int > findRows() const;
    std::vector< int > findCols() const;
    void setCoordinates( std::vector< int > a, std::vector< int > b ) { theRows = a; theCols = b; }
    std::vector< int > getRows() const { return theRows; }
    std::vector< int > getCols() const { return theCols; }

    /// Cluster width
    unsigned int findWidth() const;

    /// Single hit coordinates
    /// Average cluster coordinates
    MeasurementPoint findHitLocalCoordinates( unsigned int hitIdx ) const;
    MeasurementPoint findAverageLocalCoordinates() const;

    /// Information
    std::string print( unsigned int i = 0 ) const;

  private:
    /// Data members
    std::vector< T >  theHits;
    DetId             theDetId;
    unsigned int      theStackMember;

    std::vector< int > theRows;
    std::vector< int > theCols;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
/// NOTE: to be used with setSomething(...) methods
template< typename T >
TTCluster< T >::TTCluster()
{
  /// Set default data members
  theHits.clear();
  theDetId = 0;
  theStackMember = 0;

  theRows.clear();
  theCols.clear();
}

/// Another Constructor
template< typename T >
TTCluster< T >::TTCluster( std::vector< T > aHits,
                           DetId aDetId,
                           unsigned int aStackMember,
                           bool storeLocal )
{
  /// Set data members
  this->setHits( aHits );
  this->setDetId( aDetId );
  this->setStackMember( aStackMember );    

  theRows.clear();
  theCols.clear();
  if ( storeLocal )
  {
    this->setCoordinates( this->findRows(), this->findCols() );
  }
}

/// Destructor
template< typename T >
TTCluster< T >::~TTCluster(){}

/// Cluster width
template< >
unsigned int TTCluster< edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > >::findWidth() const;

/// Single hit coordinates
/// Average cluster coordinates
template< >
MeasurementPoint TTCluster< edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > >::findHitLocalCoordinates( unsigned int hitIdx ) const;

template< >
MeasurementPoint TTCluster< edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > >::findAverageLocalCoordinates() const;

/// Operations with coordinates stored locally
template< typename T > 
std::vector< int > TTCluster< T >::findRows() const
{
  std::vector< int > temp;
  return temp;
}

template< typename T > 
std::vector< int > TTCluster< T >::findCols() const
{
  std::vector< int > temp;
  return temp;
}

template< >
std::vector< int > TTCluster< edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > >::findRows() const;

template< >
std::vector< int > TTCluster< edm::Ref< edm::DetSetVector< PixelDigi >, PixelDigi > >::findCols() const;

/// Information
template< typename T >
std::string TTCluster< T >::print( unsigned int i ) const
{
  std::string padding("");
  for ( unsigned int j = 0; j != i; ++j )
  {
    padding+="\t";
  }

  std::stringstream output;
  output<<padding<<"TTCluster:\n";
  padding+='\t';
  output << padding << "DetId: " << theDetId.rawId() << '\n';
  output << padding << "member: " << theStackMember << ", cluster size: " << theHits.size() << '\n';
  return output.str();
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const TTCluster< T >& aTTCluster )
{
  return ( os << aTTCluster.print() );
}

#endif

