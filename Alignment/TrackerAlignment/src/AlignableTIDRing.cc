#include "Alignment/TrackerAlignment/interface/AlignableTIDRing.h"

#include <algorithm>

/// The constructor gets all components and stores them as AlignableDets
AlignableTIDRing::AlignableTIDRing( std::vector<GeomDet*>& geomDets ) 
{

  for ( std::vector<GeomDet*>::iterator iGeomDet = geomDets.begin(); 
		iGeomDet != geomDets.end(); iGeomDet++ )
    theDets.push_back( new AlignableDet(*iGeomDet) );  

  setSurface( computeSurface() );

}


/// Destructor: delete all AlignableDet objects
AlignableTIDRing::~AlignableTIDRing() 
{

  for ( std::vector<AlignableDet*>::iterator iter = theDets.begin(); 
        iter != theDets.end(); iter++)
    delete *iter;

}


/// Return all components of the ring (as Alignables)
std::vector<Alignable*> AlignableTIDRing::components() const 
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theDets.begin(), theDets.end() );
  return result;

}


/// Return AlignableDet at given index
AlignableDet &AlignableTIDRing::det(int i) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Det index (" << i << ") out of range";
  return *theDets[i];

}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableTIDRing::computeSurface() 
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average position from all components
AlignableTIDRing::PositionType AlignableTIDRing::computePosition() 
{

  // average over x, y and z

  float xx=0., yy=0., zz=0.;
  
  for ( std::vector<AlignableDet*>::const_iterator idet=theDets.begin();
		idet != theDets.end(); idet++ )
	{
	  xx += (*idet)->globalPosition().x();
	  yy += (*idet)->globalPosition().y();
	  zz += (*idet)->globalPosition().z();
	}

  xx /= static_cast<float>(theDets.size());
  yy /= static_cast<float>(theDets.size());
  zz /= static_cast<float>(theDets.size());
  
  return PositionType( xx, yy, zz );

}


/// Compute orientation from first Det
AlignableTIDRing::RotationType AlignableTIDRing::computeOrientation() 
{

  // simply take the orientation of the ring from the orientation of
  // the first Det. It works ... for a start. Later on this should
  // be taken from the universal XML description of the detector

  return  theDets.front()->globalRotation();

}


/// Printout the DetUnits in the ring
std::ostream &operator << ( std::ostream &os, const AlignableTIDRing & r )
{

  os << "    This TIDRing contains " << r.theDets.size() << " units" << std::endl;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi,r,z) =  (" << r.globalPosition().phi() << "," 
     << r.globalPosition().perp() << " , " << r.globalPosition().z();
  os << "), orientation:" << std::endl<< r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  for ( std::vector<AlignableDet*>::const_iterator idet = r.theDets.begin(); 
	idet != r.theDets.end(); idet++) 
	{
	  for (int i=0;i<(*idet)->size(); i++) 
		{
		  os << "      Det" << i << " position, phi, r: " 
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position() << ", "
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position().phi() << ", "
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position().perp() << std::endl; 
		  
		  os << "      local position, phi, r: " 
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()) << ", "
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()).phi() << ", "
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()).perp() << std::endl; 
		}
	}
  return os;

}











