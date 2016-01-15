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

StackedTrackerGeometry::StackedTrackerGeometry( const TrackerGeometry *i )
  : theTracker(i),
    theNumPartitions(0), theMaxStubs(0)
{}

StackedTrackerGeometry::StackedTrackerGeometry( const TrackerGeometry *i,
                                                const int partitionsPerRoc,
                                                const unsigned CBC3_Stubs )
  : theTracker(i),
    theNumPartitions(partitionsPerRoc),
    theMaxStubs(CBC3_Stubs)
{}

StackedTrackerGeometry::~StackedTrackerGeometry() {}

/// Methods for data members
void StackedTrackerGeometry::addStack( StackedTrackerDetUnit* aStack )
{
  theStacks.push_back( aStack );
  theStackIds.push_back( aStack->Id() );
  theMap.insert( std::make_pair( aStack->Id(), aStack ) );

  mapDetectorsToPartner.insert( std::make_pair( aStack->stackMember(0), aStack->stackMember(1) ) );
  mapDetectorsToPartner.insert( std::make_pair( aStack->stackMember(1), aStack->stackMember(0) ) );
  mapDetectorsToStack.insert( std::make_pair( aStack->stackMember(0), aStack->Id() ) );
  mapDetectorsToStack.insert( std::make_pair( aStack->stackMember(1), aStack->Id() ) );
}

const StackedTrackerDetUnit* StackedTrackerGeometry::idToStack( StackedTrackerDetId anId ) const
{
  if ( theMap.find(anId) != theMap.end() )
  {
    return theMap.find(anId)->second;
  }
  return NULL;
}

/// Association Detector/Module/Stack
DetId StackedTrackerGeometry::findPairedDetector( DetId anId ) const
{
  if ( mapDetectorsToPartner.find( anId ) != mapDetectorsToPartner.end() )
  {
    return mapDetectorsToPartner.find( anId )->second;
  }
  return DetId( 0x00000000 );
}

DetId StackedTrackerGeometry::findStackFromDetector( DetId anId ) const
{
  if ( mapDetectorsToStack.find( anId ) != mapDetectorsToStack.end() )
  {
    return mapDetectorsToStack.find( anId )->second;
  }
  return DetId( 0x00000000 );
}

/// CBC3 stuff
const int StackedTrackerGeometry::getDetUnitWindow( StackedTrackerDetId anId ) const
{
  const StackedTrackerDetUnit* theStack = this->idToStack( anId );
  if ( theStack == NULL )
  {
    return 0;
  }
  return theStack->detUnitWindow();
}

const int StackedTrackerGeometry::getASICOffset( StackedTrackerDetId anId, int asicNumber, int partitionNumber ) const
{
  const StackedTrackerDetUnit* theStack = this->idToStack( anId );
  if ( theStack == NULL )
  {
    return 999999;
  }
  return theStack->asicOffset( asicNumber, partitionNumber );
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

const bool StackedTrackerGeometry::isPSModule( StackedTrackerDetId anId ) const
{
  const GeomDetUnit* det0 = this->idToDetUnit( anId, 0 );
  const GeomDetUnit* det1 = this->idToDetUnit( anId, 1 );

  /// Find pixel pitch and topology related information
  const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
  const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
  const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );

  /// Stop if the clusters are not in the same z-segment
  int cols0 = top0->ncolumns();
  int cols1 = top1->ncolumns();
  int ratio = cols0/cols1; /// This assumes the ratio is integer!

  if ( ratio == 1 )
    return false;

  return true;
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

