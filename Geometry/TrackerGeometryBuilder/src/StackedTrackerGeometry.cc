/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
/// ////////////////////////////////////////

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

StackedTrackerGeometry::StackedTrackerGeometry( const TrackerGeometry *i ) : theTracker(i) {}
StackedTrackerGeometry::~StackedTrackerGeometry() {}

const StackedTrackerGeometry::StackContainer& StackedTrackerGeometry::stacks() const
{
  return theStacks;
}

const StackedTrackerGeometry::StackIdContainer& StackedTrackerGeometry::stackIds() const
{
  return theStackIds;
}

void StackedTrackerGeometry::addStack(StackedTrackerDetUnit* aStack)
{
  theStacks.push_back( aStack );
  theStackIds.push_back( aStack->Id() );
  theMap.insert( std::make_pair(aStack->Id(),aStack) );
}

const StackedTrackerDetUnit* StackedTrackerGeometry::idToStack( StackedTrackerDetId anId ) const
{
  if ( theMap.find(anId) != theMap.end() )
  {
    return theMap.find(anId)->second;
  }
  return NULL;
}

/// The following methods are analagous to the methods in TrackerGeomety
/// except that you pass it a stack id and an identifier to a stack member
const GeomDetUnit* StackedTrackerGeometry::idToDetUnit( StackedTrackerDetId anId , 
							unsigned int stackMemberIdentifier ) const
{
  if ( const StackedTrackerDetUnit* temp=(this->idToStack(anId)) )
  {
    return theTracker->idToDetUnit( temp->stackMember(stackMemberIdentifier) );
  }
  return NULL;
}

const GeomDet* StackedTrackerGeometry::idToDet( StackedTrackerDetId anId , 
						unsigned int stackMemberIdentifier ) const
  {
  if ( const StackedTrackerDetUnit* temp=(this->idToStack(anId)) )
  {
    return theTracker->idToDet( temp->stackMember(stackMemberIdentifier) );
  }
  return NULL;
}

Plane::PlanePointer StackedTrackerGeometry::meanPlane( StackedTrackerDetId anId ) const
{
  double x = 0.0,  y = 0.0,  z = 0.0;
  double xx = 0.0, xy = 0.0, xz = 0.0, yx = 0.0, yy = 0.0, yz = 0.0, zx = 0.0, zy = 0.0, zz = 0.0;

  const StackedTrackerDetUnit* theStack = this->idToStack( anId );

  if ( theStack == NULL )
  {
    return NULL;
  }
  else
  {
    for ( StackedTrackerDetUnit::StackContentsIterator i = theStack->theStackMembers().begin();
          i != theStack->theStackMembers().end();
          ++i )
    {
      Surface::RotationType rot = theTracker->idToDet(i->second)->rotation();
      Surface::PositionType pos = theTracker->idToDet(i->second)->position();

      x  += pos.x();
      y  += pos.y();
      z  += pos.z();
      xx += rot.xx();
      xy += rot.xy();
      xz += rot.xz();
      yx += rot.yx();
      yy += rot.yy();
      yz += rot.yz();
      zx += rot.zx();
      zy += rot.zy();
      zz += rot.zz();
    }

    float sizeInv = 1.0/theStack->theStackMembers().size();
    x  *= sizeInv;
    y  *= sizeInv;
    z  *= sizeInv;
    xx *= sizeInv;
    xy *= sizeInv;
    xz *= sizeInv;
    yx *= sizeInv;
    yy *= sizeInv;
    yz *= sizeInv;
    zx *= sizeInv;
    zy *= sizeInv;
    zz *= sizeInv;

    Surface::PositionType meanPos( x , y , z );
    Surface::RotationType meanRot( xx , xy , xz , yx , yy , yz , zx , zy , zz );

    return Plane::build( meanPos, meanRot );
  }
}

  template<>
  LocalPoint StackedTrackerGeometry::findHitLocalPosition( const L1TkCluster< edm::Ref< edm::PSimHitContainer > > *cluster,
							   unsigned int hitIdx ) const
{
  return cluster->getHits().at(hitIdx)->localPosition();
}

/// Get hit global position
/// Default template for PixelDigis in *.h
/// Specialize the template for PSimHits
  template<>
  GlobalPoint StackedTrackerGeometry::findHitGlobalPosition( const L1TkCluster< edm::Ref< edm::PSimHitContainer > > *cluster,
							     unsigned int hitIdx ) const
{
  const GeomDetUnit* geomDetUnit = idToDetUnit( cluster->getDetId(), cluster->getStackMember() );
  return geomDetUnit->surface().toGlobal( cluster->getHits().at(hitIdx)->localPosition() );
}

/// Collect MC truth
/// Default template for PixelDigis in *.h
/// Specialize the template for PSimHits
template<>
void StackedTrackerGeometry::checkSimTrack( L1TkCluster< edm::Ref< edm::PSimHitContainer > > *cluster,
					    edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  thePixelDigiSimLinkHandle,
					    edm::Handle<edm::SimTrackContainer>   simTrackHandle ) const
{
  /// Loop over all the hits composing the L1TkCluster
  std::vector< edm::Ref< edm::PSimHitContainer > > hits=cluster->getHits();
  for ( unsigned int i = 0; i < hits.size(); i++ ) {

    /// Get SimTrack Id and type
    unsigned int curSimTrkId = hits.at(i)->trackId();

    /// This version of the collection of the SimTrack ID and PDG
    /// may not be fast and optimal, but is safer since the
    /// SimTrack ID is shifted by 1 wrt the index in the vector,
    /// and this may not be so true on a general basis...
    bool foundSimTrack = false;
    for ( unsigned int j = 0; j < simTrackHandle->size() && !foundSimTrack; j++ )
      {
        if ( simTrackHandle->at(j).trackId() == curSimTrkId )
	  {
	    foundSimTrack = true;
	    edm::Ptr< SimTrack > testSimTrack( simTrackHandle, j );
	    cluster->addSimTrack( testSimTrack );
	  }
      }
    if ( !foundSimTrack )
      {
	edm::Ptr< SimTrack > testSimTrack;
        cluster->addSimTrack( testSimTrack );
      }
  } /// End of Loop over all the hits composing the L1TkCluster
}



