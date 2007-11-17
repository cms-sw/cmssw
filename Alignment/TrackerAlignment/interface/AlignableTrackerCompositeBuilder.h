#ifndef Alignment_TrackerAlignment_AlignableTrackerCompositeBuilder_H
#define Alignment_TrackerAlignment_AlignableTrackerCompositeBuilder_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Tracker components
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include <vector>

typedef GeometricDet::ConstGeometricDetContainer _DetContainer;

/// A Builder class for composite alignables
///
/// The AlignableTrackerCompositeBuilder looks for the GeomDets
/// corresponding to the list of given GeometricDets, and returns
/// and Alignable made of these GeomDets. If the given GeometricDets
/// do not have associated GeomDets, this class will extract them
/// from the sub-components.

template<class C>
class AlignableTrackerCompositeBuilder
{

public:

  /// Constructor
  AlignableTrackerCompositeBuilder() {}

  /// Desctructor
  ~AlignableTrackerCompositeBuilder() {}

  /// Build alignable object from list of GeomDets
  C* buildAlignable( _DetContainer Dets,
					 const TrackerGeometry* geomDetGeometry
					 ) const;

private:
  
  // Returns a list of GeometricDets that correspond to GeomDets
  _DetContainer extractGeomDets( _DetContainer Dets ) const;

};


//--------------------------------------------------------------------------------------------------
template<class C> 
C* AlignableTrackerCompositeBuilder<C>::buildAlignable( _DetContainer Dets,
														const TrackerGeometry* geomDetGeometry
														) const
{
  
  std::vector<const GeomDet*> geomDets;
  
  _DetContainer m_Dets = extractGeomDets( Dets );

  for ( _DetContainer::iterator iDet = m_Dets.begin(); iDet != m_Dets.end(); iDet++ )
	{
	  // Cast away const here
	  const GeomDet* geomDet = geomDetGeometry->idToDet((*iDet)->geographicalID());
	  geomDets.push_back( geomDet );
	}
  
  return new C( geomDets );

}


//--------------------------------------------------------------------------------------------------
template<class C> 
_DetContainer AlignableTrackerCompositeBuilder<C>::extractGeomDets( _DetContainer Dets ) const
{

  // Check if given GeometricDets have corresponding GeomDets
  // (only GluedDets (=mergedDet) and DetUnits are in this case)

  _DetContainer result;

  for ( _DetContainer::iterator iDet = Dets.begin(); iDet != Dets.end(); iDet++ )
	{
	  if ( (*iDet)->type() == GeometricDet::mergedDet
		   || (*iDet)->type() == GeometricDet::DetUnit )
		result.push_back(*iDet);
	  else
		{
		  _DetContainer tmpDets = extractGeomDets( (*iDet)->components() );
		  result.insert( result.end(), tmpDets.begin(), tmpDets.end() );
		}
	}

  return result;

}



#endif

