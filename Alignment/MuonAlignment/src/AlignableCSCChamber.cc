#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"


/// The constructor gets all components and stores them as AlignableDets
AlignableCSCChamber::AlignableCSCChamber( GeomDet* geomDet  ) : AlignableComposite( geomDet )
{

  // Retrieve components
  if ( geomDet->components().size() ) 
	{
	  std::vector<const GeomDet*> m_SuperLayers = geomDet->components();
	  for ( std::vector<const GeomDet*>::iterator iGeomDet = m_SuperLayers.begin(); 
			iGeomDet != m_SuperLayers.end(); iGeomDet++ )
		{
		  GeomDet* tmpGeomDet = const_cast<GeomDet*>(*iGeomDet);
		  theDets.push_back( new AlignableDet(tmpGeomDet) );
		}
	}

  setSurface( computeSurface() );

}


      

/// Destructor: delete all AlignableDet objects
AlignableCSCChamber::~AlignableCSCChamber() 
{

  // Properly delete all elements of the vector (they are pointers!)
  for ( std::vector<AlignableDet*>::iterator iter = theDets.begin(); 
		iter != theDets.end(); iter++)
    delete *iter;

}



/// Return all components of the chamber (as Alignables)
std::vector<Alignable*> AlignableCSCChamber::components() const 
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theDets.begin(), theDets.end() );
  return result;
  
}






/// Return Alignable CSC Chamber at given index
AlignableDet& AlignableCSCChamber::det( int i )
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Det index (" << i << ") out of range";

  return *theDets[i];
  
}




/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableCSCChamber::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average position from geomDet
AlignableCSCChamber::PositionType AlignableCSCChamber::computePosition() 
{
  
  return theGeomDet->position();

}


/// Compute orientation from geomDet
AlignableCSCChamber::RotationType AlignableCSCChamber::computeOrientation() 
{
  
  return theGeomDet->rotation();

}


/// Return length from surface
float AlignableCSCChamber::length() const 
{

  return theGeomDet->surface().bounds().length();

}




/// Printout the DetUnits in the CSC chamber
std::ostream &operator << ( std::ostream &os, const AlignableCSCChamber & r )
{

  os << "    This CSCChamber contains " << r.theDets.size() << " units" << std::endl ;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," 
     << r.globalPosition().perp() << "," << r.globalPosition().z();
  os << "), orientation:" << std::endl<< r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  for ( std::vector<AlignableDet*>::const_iterator idet = r.theDets.begin(); 
		idet != r.theDets.end(); idet++) 
	{
	  for ( int i=0; i<(*idet)->size();i++) 
		{
		  os << "     Det position, phi, r: " 
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position() << " , "
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position().phi() << " , "
			 << ((*idet)->geomDetUnit(i).geomDetUnit())->position().perp() << std::endl; 
		  os << "     local  position, phi, r: " 
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()) 
			 << " , "
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()).phi() << " , "
			 << r.surface().toLocal(((*idet)->geomDetUnit(i).geomDetUnit())->position()).perp() << std::endl; 
		}
	}
  return os;

}
