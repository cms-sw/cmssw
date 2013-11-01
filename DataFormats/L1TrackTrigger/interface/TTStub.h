/*! \class   TTStub
 *  \brief   Class to store the L1 Track Trigger stubs
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_FORMAT_H
#define L1_TRACK_TRIGGER_STUB_FORMAT_H

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

template< typename T >
class TTStub
{
  public:
    /// Constructors
    TTStub();
    TTStub( DetId aDetId );

    /// Destructor
    ~TTStub();

    /// Data members:   getABC( ... )
    /// Helper methods: findABC( ... )

    /// Clusters composing the Stub
    std::vector< edm::Ptr< TTCluster< T > > > getClusterPtrs() const                                 { return theClusters; }
    const edm::Ptr< TTCluster< T > >&         getClusterPtr( unsigned int hitIdentifier ) const;
    void                                      addClusterPtr( edm::Ptr< TTCluster< T > > aTTCluster ) { theClusters.push_back( aTTCluster ); }

    /// Detector element
    DetId getDetId() const         { return theDetId; }
    void  setDetId( DetId aDetId ) { theDetId = aDetId; }

    /// Trigger information
    double getTriggerDisplacement() const;              /// In FULL-STRIP units! (hence, not implemented herein)
    void   setTriggerDisplacement( int aDisplacement ); /// In HALF-STRIP units!
    double getTriggerOffset() const;         /// In FULL-STRIP units! (hence, not implemented herein)
    void   setTriggerOffset( int anOffset ); /// In HALF-STRIP units!

    /// CBC3-style trigger information
    /// for sake of simplicity, these methods are
    /// slightly out of the getABC(...)/findABC(...) rule
    double getTriggerPosition() const; /// In FULL-STRIP units!
    double getTriggerBend() const;     /// In FULL-STRIP units!

    /// Information
    std::string print( unsigned int i = 0 ) const;

  private:
    /// Data members
    DetId                                     theDetId;
    std::vector< edm::Ptr< TTCluster< T > > > theClusters;
    int                                       theDisplacement;
    int                                       theOffset;

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Default Constructor
template< typename T >
TTStub< T >::TTStub()
{
  /// Set default data members
  theDetId = 0;
  theClusters.clear();
  theDisplacement = 999999;
  theOffset = 0;
}

/// Another Constructor
template< typename T >
TTStub< T >::TTStub( DetId aDetId )
{
  /// Set data members
  this->setDetId( aDetId );

  /// Set default data members
  theClusters.clear();
  theDisplacement = 999999;
  theOffset = 0;
}

/// Destructor
template< typename T >
TTStub< T >::~TTStub(){}

/// Get the Pointer to a Cluster
template< typename T >
const edm::Ptr< TTCluster< T > >& TTStub< T >::getClusterPtr( unsigned int hitIdentifier ) const
{
  /// Look for the TTCluster with the stack member corresponding to the argument
  typename std::vector< edm::Ptr< TTCluster< T > > >::const_iterator clusIter;
  for ( clusIter = theClusters.begin();
        clusIter != theClusters.end();
        ++clusIter )
  {
    if ( (*clusIter)->getStackMember() == hitIdentifier )
    {
      return *clusIter;
    }
  }

  /// In case no TTCluster is found, return a NULL edm::Ptr
  /// (hopefully code doesn't reach this point)
  edm::Ptr< TTCluster< T > >* tmpCluPtr = new edm::Ptr< TTCluster< T > >();
  return *tmpCluPtr;
}

/// Trigger info
template< typename T >
double TTStub< T >::getTriggerDisplacement() const { return 0.5*theDisplacement; }

template< typename T >
void TTStub< T >::setTriggerDisplacement( int aDisplacement ) { theDisplacement = aDisplacement; }

template< typename T >
double TTStub< T >::getTriggerOffset() const { return 0.5*theOffset; }

template< typename T >
void TTStub< T >::setTriggerOffset( int anOffset ) { theOffset = anOffset; }

/// CBC3-style trigger info
template< typename T >
double TTStub< T >::getTriggerPosition() const
{
  return this->getClusterPtr(0)->findAverageLocalCoordinates().x();
}

template< typename T >
double TTStub< T >::getTriggerBend() const { return 0.5*( theDisplacement - theOffset ); }

/// Information
template< typename T >
std::string TTStub< T >::print( unsigned int i ) const
{
  std::string padding("");
  for ( unsigned int j = 0; j != i; ++j )
  {
    padding+="\t";
  }

  std::stringstream output;
  output<<padding<<"TTStub:\n";
  padding+='\t';
  output << padding << "DetId: " << theDetId.rawId() << '\n';
  unsigned int iClu = 0;
  typename std::vector< edm::Ptr< TTCluster< T > > >::const_iterator clusIter;
  for ( clusIter = theClusters.begin();
        clusIter!= theClusters.end();
        ++clusIter )
  {
    output << padding << "cluster: " << iClu++ << ", member: " << (*clusIter)->getStackMember();
    output << ", cluster size: " << (*clusIter)->getHits().size() << '\n';
  }

  return output.str();
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const TTStub< T >& aTTStub ) { return ( os << aTTStub.print() ); }

#endif

