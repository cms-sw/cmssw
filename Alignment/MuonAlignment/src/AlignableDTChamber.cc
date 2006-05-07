#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"


/// The constructor gets all components and stores them as AlignableDets
AlignableDTChamber::AlignableDTChamber( std::vector<GeomDet*>& geomDets  ) 
{

  for ( std::vector<GeomDet*>::iterator iGeomDet=geomDets.begin(); 
		iGeomDet != geomDets.end(); iGeomDet++ )
	theDets.push_back( new AlignableDet(*iGeomDet) );

  setSurface( computeSurface() );

}


      

/// Destructor: delete all AlignableDet objects
AlignableDTChamber::~AlignableDTChamber() 
{

  // Properly delete all elements of the vector (they are pointers!)
  for ( std::vector<AlignableDet*>::iterator iter = theDets.begin(); 
		iter != theDets.end(); iter++)
    delete *iter;

}



/// Return all components of the chamber (as Alignables)
std::vector<Alignable*> AlignableDTChamber::components() const 
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theDets.begin(), theDets.end() );
  return result;
  
}






/// Return Alignable DT Chamber at given index
AlignableDet& AlignableDTChamber::det( int i )
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Det index (" << i << ") out of range";

  return *theDets[i];
  
}




/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableDTChamber::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


/// Compute average position from all components
AlignableDTChamber::PositionType AlignableDTChamber::computePosition() 
{
  
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


/// Compute orientation from position
AlignableDTChamber::RotationType AlignableDTChamber::computeOrientation() {

  // Force the z-axis along the r-phi direction
  //           y-axis along the global z direction
  //           x-axis such that you get a right handed system
  // (x and y in the (nominal) plane of the rod)

  GlobalVector vec = ( this->computePosition() - GlobalPoint(0,0,0) );
  GlobalVector vecrphi = GlobalVector(vec.x(), vec.y(),0.).unit();
  GlobalVector lxaxis = GlobalVector(0.,0.,1.).cross(vecrphi);
  RotationType orientation(
						   lxaxis.x(), lxaxis.y(), lxaxis.z(), 
						   0.,          0.,           1.,
						   vecrphi.x(), vecrphi.y(), vecrphi.z()
						   );

  return orientation;

}



/// Return length calculated from components
float AlignableDTChamber::length() const 
{

  float zz, zmin=+10000., zmax=-10000.;
  for ( std::vector<AlignableDet*>::const_iterator idet=theDets.begin();
		idet != theDets.end(); idet++ )
	{
	  zz = (*idet)->globalPosition().z();
	  if (zz < zmin) zmin = zz;
	  if (zz > zmax) zmax = zz;
	}

  return zmax-zmin;

}




/// Printout the DetUnits in the DT chamber
std::ostream &operator << ( std::ostream &os, const AlignableDTChamber & r )
{

  os << "    This DTChamber contains " << r.theDets.size() << " units" << std::endl ;
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
