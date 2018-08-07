/*! \class   TTStub
 *  \brief   Class to store the L1 Track Trigger stubs
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than Phase2TrackerDigis
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
#include "DataFormats/Common/interface/DetSetVectorNew.h"

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
    const edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > >&         getClusterRef( unsigned int hitIdentifier ) const;
    void addClusterRef( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aTTCluster );

    /// Detector element
    DetId getDetId() const         { return theDetId; }
    void  setDetId( DetId aDetId ) { theDetId = aDetId; }

    /// Trigger information
    double getTriggerDisplacement() const;              /// In FULL-STRIP units! (hence, not implemented herein)
    void   setTriggerDisplacement( int aDisplacement ); /// In HALF-STRIP units!
    double getTriggerOffset() const;         /// In FULL-STRIP units! (hence, not implemented herein)
    void   setTriggerOffset( int anOffset ); /// In HALF-STRIP units!
    double getRealTriggerOffset() const;         /// In FULL-STRIP units! (hence, not implemented herein)
    void   setRealTriggerOffset( float anOffset ); /// In HALF-STRIP units!



    /// CBC3-style trigger information
    /// for sake of simplicity, these methods are
    /// slightly out of the getABC(...)/findABC(...) rule
    double getTriggerPosition() const; /// In FULL-STRIP units!
    double getTriggerBend() const;     /// In FULL-STRIP units!
    double getHardwareBend() const; /// In FULL-STRIP units!
    void   setHardwareBend( float aBend ); /// In HALF-STRIP units!

    /// Information
    std::string print( unsigned int i = 0 ) const;

  private:
    /// Data members
    DetId theDetId;
    edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > theClusterRef0;
    edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > theClusterRef1;
    int theDisplacement;
    int theOffset;
    float theRealOffset;
    float theHardwareBend;

    static constexpr float dummyBend = 999999; // Dumy value should be away from potential bends
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
  theDisplacement = dummyBend;
  theOffset = 0;
  theRealOffset = 0;
  theHardwareBend = dummyBend;
}

/// Another Constructor
template< typename T >
TTStub< T >::TTStub( DetId aDetId )
{
  /// Set data members
  this->setDetId( aDetId );

  /// Set default data members
  theDisplacement = dummyBend;
  theOffset = 0;
  theRealOffset = 0;
  theHardwareBend = dummyBend;
}

/// Destructor
template< typename T >
TTStub< T >::~TTStub(){}

/// Get the Reference to a Cluster
template< typename T >
const edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > >& TTStub< T >::getClusterRef( unsigned int hitIdentifier ) const
{
  return hitIdentifier==0 ? theClusterRef0 : theClusterRef1;
}

template< typename T >
void TTStub< T >::addClusterRef( edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > aTTCluster )
{
  if(aTTCluster->getStackMember() == 0) theClusterRef0 = aTTCluster;
  else if (aTTCluster->getStackMember() == 1) theClusterRef1 = aTTCluster;
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

template< typename T >
double TTStub< T >::getRealTriggerOffset() const { return 0.5*theRealOffset; }

template< typename T >
void TTStub< T >::setRealTriggerOffset( float anOffset ) { theRealOffset = anOffset; }

template< typename T >
void TTStub< T >::setHardwareBend( float aBend ) { theHardwareBend = aBend; }


/// CBC3-style trigger info
template< typename T >
double TTStub< T >::getTriggerPosition() const
{
  return this->getClusterRef(0)->findAverageLocalCoordinates().x();
}

template< typename T >
double TTStub< T >::getTriggerBend() const
{
  if ( theDisplacement == dummyBend )
    return theDisplacement;

  return 0.5*( theDisplacement - theOffset );
}

template< typename T >
double TTStub< T >::getHardwareBend() const
{
  if ( theHardwareBend == dummyBend )
    return this->getTriggerBend(); // If not set make it transparent

  return theHardwareBend;
}


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
  output << padding << "DetId: " << theDetId.rawId() << ", position: " << this->getTriggerPosition();
  output << ", bend: " << this->getTriggerBend() << '\n';
  output << ", hardware bend: " << this->getHardwareBend() << '\n';
  output << padding << "cluster 0: address: " << theClusterRef0.get();
  output << ", cluster size: " << theClusterRef0->getHits().size() << '\n';
  output << padding << "cluster 1: address: " << theClusterRef1.get();
  output << ", cluster size: " << theClusterRef1->getHits().size() << '\n';
  return output.str();
}

template< typename T >
std::ostream& operator << ( std::ostream& os, const TTStub< T >& aTTStub ) { return ( os << aTTStub.print() ); }

#endif

