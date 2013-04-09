
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_GEOMETRY_H
#define STACKED_TRACKER_GEOMETRY_H

#include <typeinfo>

#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <vector>
#include <ext/hash_map>

class StackedTrackerDetUnit;
class GeomDet;
class GeomDetUnit;
class TrackerGeometry;

class StackedTrackerGeometry {
public:
  typedef	std::vector<StackedTrackerDetUnit*>	StackContainer;
  typedef	std::vector<StackedTrackerDetUnit*>::const_iterator	StackContainerIterator;
  typedef	std::vector<StackedTrackerDetId>		StackIdContainer;
  typedef	std::vector<StackedTrackerDetId>::const_iterator		StackIdContainerIterator;
  typedef  __gnu_cxx::hash_map< unsigned int, StackedTrackerDetUnit*> mapIdToStack;

  StackedTrackerGeometry( const TrackerGeometry *i );  
  virtual ~StackedTrackerGeometry();  

  const StackContainer&		stacks()	const;
  const StackIdContainer&	stackIds()	const;
 
  void	addStack(StackedTrackerDetUnit *aStack);

  const StackedTrackerDetUnit*		idToStack( StackedTrackerDetId anId )	const;


//analagous to the methods in TrackerGeomety except that you pass it a stack id and an identifier to a stack member
  const GeomDetUnit*       idToDetUnit( StackedTrackerDetId anId , unsigned int stackMemberIdentifier ) const;
  const GeomDet*           idToDet( StackedTrackerDetId anId , unsigned int stackMemberIdentifier )     const;


//helper functions
  Plane::PlanePointer meanPlane(StackedTrackerDetId anId) const;


private:

  const TrackerGeometry* theTracker;

  StackContainer	theStacks;
  StackIdContainer	theStackIds;

  mapIdToStack    theMap;

};


#endif

