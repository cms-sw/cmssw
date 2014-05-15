/*! \class   TTTrack
 *  \brief   Class to store the L1 Track Trigger tracks
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_FORMAT_H
#define L1_TRACK_TRIGGER_TRACK_FORMAT_H

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"

template< typename T >
class TTTrack
{
  private:
    /// Data members
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > >
                 theStubRefs;
    GlobalVector theMomentum;
    GlobalPoint  thePOCA;
    double       theRInv;
    unsigned int theSector;
    unsigned int theWedge;
    double       theChi2;
    unsigned int theFitParNo;

  public:
    /// Constructors
    TTTrack();
    TTTrack( std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > aStubs );

    /// Destructor
    ~TTTrack();

    /// Track components
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > getStubRefs() const        { return theStubRefs; }
    void addStubRef( edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > aStub )                  { theStubRefs.push_back( aStub ); }
    void setStubRefs( std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > aStubs ) { theStubRefs = aStubs; }

    /// Track momentum
    GlobalVector getMomentum() const                   { return theMomentum; }
    void         setMomentum( GlobalVector aMomentum ) { theMomentum = aMomentum; }

    /// Track curvature
    double getRInv() const         { return theRInv; }
    void   setRInv( double aRInv ) { theRInv = aRInv; }

    /// POCA
    GlobalPoint getPOCA() const              { return thePOCA; }
    void        setPOCA( GlobalPoint aPOCA ) { thePOCA = aPOCA; }

    /// Sector
    unsigned int getSector() const                 { return theSector; }
    void         setSector( unsigned int aSector ) { theSector = aSector; }
    unsigned int getWedge() const                  { return theWedge; }
    void         setWedge( unsigned int aWedge )   { theWedge = aWedge; }

    /// Chi2
    double       getChi2() const                       { return theChi2; }
    double       getChi2Red() const;
    void         setChi2( double aChi2 )               { theChi2 = aChi2; }
    unsigned int getFitParNo() const                   { return theFitParNo; }
    void         setFitParNo( unsigned int aFitParNo ) { theFitParNo = aFitParNo; }

/*
    /// Superstrip
    /// Here to prepare inclusion of AM L1 Track finding
    uint32_t getSuperStrip() const { return 0; }
*/
    /// Duplicate identification
    bool isTheSameAs( TTTrack< T > aTrack ) const;

    /// Additional quality criteria
    bool hasStubInBarrel( unsigned int aLayer ) const;

    /// Information
    std::string print( unsigned int i = 0 ) const;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */
 
/// Default Constructor
template< typename T >
TTTrack< T >::TTTrack()
{
  theStubRefs.clear();
  theMomentum = GlobalVector(0.0,0.0,0.0);
  theRInv     = 0;
  thePOCA     = GlobalPoint(0.0,0.0,0.0);
  theSector   = 0;
  theWedge    = 0;
  theChi2     = 0;
  theFitParNo = 0;
}

/// Another Constructor
template< typename T >
//TTTrack< T >::TTTrack( std::vector< edm::Ptr< TTStub< T > > > aStubs )
TTTrack< T >::TTTrack( std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > aStubs )
{
  theStubRefs = aStubs;
  theMomentum = GlobalVector(0.0,0.0,0.0);
  thePOCA     = GlobalPoint(0.0,0.0,0.0);
  theRInv     = 0;
  theSector   = 0;
  theWedge    = 0;
  theChi2     = 0;
  theFitParNo = 0;
}

/// Destructor
template< typename T >
TTTrack< T >::~TTTrack(){}

/// Chi2
template< typename T >
double TTTrack< T >::getChi2Red() const
{
  return theChi2/( 2*theStubRefs.size() - this->getFitParNo() );
}

/// Duplicate identification
template< typename T>
bool TTTrack< T >::isTheSameAs( TTTrack< T > aTrack ) const
{
  /// Take the other stubs
//  std::vector< edm::Ptr< TTStub< T > > > otherStubPtrs = aTrack.getStubPtrs();
  std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > otherStubRefs = aTrack.getStubRefs();

  /// Count shared stubs
  unsigned int nShared = 0;
  for ( unsigned int i = 0; i < theStubRefs.size() && nShared < 2; i++)
  {
    for ( unsigned int j = 0; j < otherStubRefs.size() && nShared < 2; j++)
    {
      if ( theStubRefs.at(i) == otherStubRefs.at(j) )
      {
        nShared++;
      }
    }
  }

  /// Same track if 2 shared stubs
  return ( nShared > 1 );
}

/// Quality criteria: does it have a Stub in a specific Barrel Layer?
template< typename T >
bool TTTrack< T >::hasStubInBarrel( unsigned int aLayer ) const
{
  for ( unsigned int i = 0; i < theStubRefs.size(); i++)
  {
    StackedTrackerDetId thisDetId( theStubRefs.at(i)->getDetId() );
    if ( thisDetId.isBarrel() && thisDetId.iLayer() == aLayer )
    {
      return true;
    }
  }

  return false;
}

/// Information
template< typename T >
std::string TTTrack< T >::print( unsigned int i ) const
{
  std::string padding("");
  for ( unsigned int j = 0; j != i; ++j )
  {
    padding+="\t";
  }

  std::stringstream output;
  output<<padding<<"TTTrack:\n";
  padding+='\t';
  output << '\n';
  unsigned int iStub = 0;

  typename std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > >::const_iterator stubIter;
  for ( stubIter = theStubRefs.begin();
        stubIter!= theStubRefs.end();
        ++stubIter )
  {
    output << padding << "stub: " << iStub++ << ", DetId: " << ((*stubIter)->getDetId()).rawId() << '\n';
  }

  //output << ", z-vertex: " << thePOCA.z() << " (cm), transverse momentum " << theMomentum.perp() << " (GeV/c)";
  //output << ", red. chi2 " << this->getChi2Red() << '\n';

  return output.str();
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const TTTrack< T >& aTTTrack ) { return ( os << aTTTrack.print() ); }

#endif

