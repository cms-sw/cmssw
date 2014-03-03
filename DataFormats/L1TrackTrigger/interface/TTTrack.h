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
#include "FWCore/MessageLogger/interface/MessageLogger.h"


template< typename T >
class TTTrack
{
  private:
    /// Data members
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > >
                 theStubRefs;
    GlobalVector theMomentum4Par;
    GlobalVector theMomentum5Par;
    GlobalPoint  thePOCA4Par;
    GlobalPoint  thePOCA5Par;
    double       theRInv4Par;
    double       theRInv5Par;
    unsigned int theSector;
    unsigned int theWedge;
    double       theStubPtConsistency4Par;
    double       theStubPtConsistency5Par;
    double       theChi24Par;
    double       theChi25Par;
    bool         valid4ParFit;
    bool         valid5ParFit;

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
    GlobalVector getMomentum(unsigned int nPar=4) const;
    void         setMomentum( GlobalVector aMomentum, unsigned int nPar=5);

    /// Track curvature
    double getRInv(unsigned int nPar=4) const;
    void   setRInv( double aRInv, unsigned int nPar=5 );

    /// POCA
    GlobalPoint getPOCA(unsigned int nPar=4) const;
    void        setPOCA( GlobalPoint aPOCA, unsigned int nPar=5 );

    /// Sector
    unsigned int getSector() const                 { return theSector; }
    void         setSector( unsigned int aSector ) { theSector = aSector; }
    unsigned int getWedge() const                  { return theWedge; }
    void         setWedge( unsigned int aWedge )   { theWedge = aWedge; }

    /// Chi2
    double       getChi2(unsigned int nPar=4) const;
    double       getChi2Red(unsigned int nPar=4) const;
    void         setChi2( double aChi2, unsigned int nPar=5 );

    /// Stub Pt consistency
    double       getStubPtConsistency(unsigned int nPar=4) const;
    void         setStubPtConsistency( double aPtConsistency, unsigned int nPar=5 );

    void         setFitParNo( unsigned int aFitParNo ) { return; }

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

 private:

    bool checkValidArgs(unsigned int nPar, std::string fcn) const;
    bool checkValidArgsForSet(unsigned int nPar, std::string fcn) const;
  

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
  theMomentum4Par = GlobalVector(0.0,0.0,0.0);
  theMomentum5Par = GlobalVector(0.0,0.0,0.0);
  theRInv4Par     = 0.0;
  theRInv5Par     = 0.0;
  thePOCA4Par     = GlobalPoint(0.0,0.0,0.0);
  thePOCA5Par     = GlobalPoint(0.0,0.0,0.0);
  theSector       = 0;
  theWedge        = 0;
  theChi24Par     = 0.0;
  theChi25Par     = 0.0;
  theStubPtConsistency4Par = 0.0;
  theStubPtConsistency5Par = 0.0;
  valid4ParFit    = false;
  valid5ParFit    = false;
}

/// Another Constructor
template< typename T >
//TTTrack< T >::TTTrack( std::vector< edm::Ptr< TTStub< T > > > aStubs )
TTTrack< T >::TTTrack( std::vector< edm::Ref< edmNew::DetSetVector< TTStub< T > >, TTStub< T > > > aStubs )
{
  theStubRefs = aStubs;
  theMomentum4Par = GlobalVector(0.0,0.0,0.0);
  theMomentum5Par = GlobalVector(0.0,0.0,0.0);
  theRInv4Par     = 0.0;
  theRInv5Par     = 0.0;
  thePOCA4Par     = GlobalPoint(0.0,0.0,0.0);
  thePOCA5Par     = GlobalPoint(0.0,0.0,0.0);
  theSector       = 0;
  theWedge        = 0;
  theChi24Par     = 0.0;
  theChi25Par     = 0.0;
  theStubPtConsistency4Par = 0.0;
  theStubPtConsistency5Par = 0.0;
  valid4ParFit    = false;
  valid5ParFit    = false;
}

/// Destructor
template< typename T >
TTTrack< T >::~TTTrack(){}

template< typename T >
void TTTrack< T >::setMomentum( GlobalVector aMomentum, unsigned int nPar ) {

  if (!checkValidArgsForSet(nPar,"setMomentum")){
    return;
  }

  if (nPar==4) {
    valid4ParFit = true;
    theMomentum4Par=aMomentum;
  }

  if (nPar==5) {
    valid5ParFit = true;
    theMomentum5Par=aMomentum;
  }

  return;


} 


template< typename T >
GlobalVector TTTrack< T >::getMomentum(unsigned int nPar) const{

  if (!checkValidArgs(nPar,"getMomentum")){
    return GlobalVector(0.0,0.0,0.0);
  }

  if (nPar==4) {
    return theMomentum4Par;
  }

  if (nPar==5) {
    return theMomentum5Par;
  }

  return GlobalVector(0.0,0.0,0.0);


} 


template< typename T >
void TTTrack< T >::setRInv(double aRInv, unsigned int nPar) {

  if (!checkValidArgsForSet(nPar,"setRInv")){
    return;
  }

  if (nPar==4) {
    valid4ParFit = true;
    theRInv4Par=aRInv;
  }

  if (nPar==5) {
    valid5ParFit = true;
    theRInv5Par=aRInv;
  }

  return;

}


template< typename T >
double TTTrack< T >::getRInv(unsigned int nPar) const {

  if (!checkValidArgs(nPar,"getRInv")){
    return 0.0;
  }

  if (nPar==4) {
    return theRInv4Par;
  }

  if (nPar==5) {
    return theRInv5Par;
  }

  return 0.0;

}


template< typename T >
void TTTrack< T >::setPOCA(GlobalPoint aPOCA, unsigned int nPar){

  if (!checkValidArgsForSet(nPar,"setPOCA")){
    return;
  }

  if (nPar==4) {
    valid4ParFit = true;
    thePOCA4Par=aPOCA;
  }

  if (nPar==5) {
    valid5ParFit = true;
    thePOCA5Par=aPOCA;
  }

  return;

}

template< typename T >
GlobalPoint TTTrack< T >::getPOCA(unsigned int nPar) const
{

  if (!checkValidArgs(nPar,"getPOCA")){
    return GlobalPoint(0.0,0.0,0.0);
  }

  if (nPar==4) {
    return thePOCA4Par;
  }

  if (nPar==5) {
    return thePOCA5Par;
  }

  return GlobalPoint(0.0,0.0,0.0);

}

/// Chi2 
template< typename T >
void TTTrack< T >::setChi2(double aChi2, unsigned int nPar) {

  if (!checkValidArgsForSet(nPar,"setChi2")){
    return;
  }

  if (nPar==4) {
    valid4ParFit = true;
    theChi24Par=aChi2;
  }

  if (nPar==5) {
    valid5ParFit = true;
    theChi25Par=aChi2;
  }

  return;

}



/// Chi2 
template< typename T >
double TTTrack< T >::getChi2(unsigned int nPar) const
{

  if (!checkValidArgs(nPar,"getChi2")){
    return 0.0;
  }

  if (nPar==4) {
    return theChi24Par;
  }

  if (nPar==5) {
    return theChi25Par;
  }

  return 0.0;

}



/// Chi2 reduced
template< typename T >
double TTTrack< T >::getChi2Red(unsigned int nPar) const
{

  if (!checkValidArgs(nPar,"getChi2Red")){
    return 0.0;
  }

  if (nPar==4) {
    return theChi24Par/( 2*theStubRefs.size() - 4 );
  }

  if (nPar==5) {
    return theChi25Par/( 2*theStubRefs.size() - 5 );
  }

  return 0.0;

}


/// StubPtConsistency 
template< typename T >
void TTTrack< T >::setStubPtConsistency(double aStubPtConsistency, unsigned int nPar) {

  if (!checkValidArgsForSet(nPar,"setStubPtConsistency")){
    return;
  }

  if (nPar==4) {
    valid4ParFit = true;
    theStubPtConsistency4Par=aStubPtConsistency;
  }

  if (nPar==5) {
    valid5ParFit = true;
    theStubPtConsistency5Par=aStubPtConsistency;
  }

  return;

}



/// StubPtConsistency 
template< typename T >
double TTTrack< T >::getStubPtConsistency(unsigned int nPar) const
{

  if (!checkValidArgs(nPar,"getStubPtConsistency")){
    return 0.0;
  }

  if (nPar==4) {
    return theStubPtConsistency4Par;
  }

  if (nPar==5) {
    return theStubPtConsistency5Par;
  }

  return 0.0;

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
bool TTTrack< T >::checkValidArgs(unsigned int nPar, std::string fcn) const {
      
  if (!(nPar==4||nPar==5)) {
    edm::LogError("TTTrack") << " In method "<<fcn<<
      " called with nPar="<<nPar<<std::endl;
    return false;
  }

  if ((nPar==4)&&!valid4ParFit) {
    edm::LogError("TTTrack") << " In method "<<fcn<<
      " called with nPar="<<nPar<<" but no valid 4 parameter fit"<<std::endl;
    return false;
  }

  if ((nPar==5)&&!valid5ParFit) {
    edm::LogError("TTTrack") << " In method "<<fcn<<
      " called with nPar="<<nPar<<" but no valid 5 parameter fit"<<std::endl;
    return false;
  }

  return true;
  
}
template< typename T >
bool TTTrack< T >::checkValidArgsForSet(unsigned int nPar, std::string fcn) const {
      
  if (!(nPar==4||nPar==5)) {
    edm::LogError("TTTrack") << " In method "<<fcn<<
      " called with nPar="<<nPar<<std::endl;
    return false;
  }

  return true;
  
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const TTTrack< T >& aTTTrack ) { return ( os << aTTTrack.print() ); }


#endif

