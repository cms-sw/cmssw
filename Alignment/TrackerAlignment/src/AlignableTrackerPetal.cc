#include "Alignment/TrackerAlignment/interface/AlignableTrackerPetal.h"

#include <algorithm>

//__________________________________________________________________________________________________
/// The constructor gets all components and stores them as AlignableDets
AlignableTrackerPetal::AlignableTrackerPetal( GeomDetContainer& geomDets) 
{

  for ( GeomDetContainer::iterator iGeomDet = geomDets.begin(); 
		iGeomDet != geomDets.end(); iGeomDet++ )
    theDets.push_back( new AlignableDet(*iGeomDet) );  

  setSurface( computeSurface() );

}



//__________________________________________________________________________________________________
AlignableTrackerPetal::~AlignableTrackerPetal() 
{

  for ( std::vector<AlignableDet*>::iterator iter = theDets.begin(); 
        iter != theDets.end(); iter++ )
    delete *iter;

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableTrackerPetal::components() const 
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theDets.begin(), theDets.end() );
  return result;

}


//__________________________________________________________________________________________________
AlignableDet& AlignableTrackerPetal::det(int i) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Det index (" << i << ") out of range";
  return *theDets[i];

}


//__________________________________________________________________________________________________
AlignableSurface AlignableTrackerPetal::computeSurface() 
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//__________________________________________________________________________________________________
AlignableTrackerPetal::PositionType AlignableTrackerPetal::computePosition() {
  

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


//__________________________________________________________________________________________________
AlignableTrackerPetal::RotationType AlignableTrackerPetal::computeOrientation() {

  // simply take the orientation of the petal from the orientation of
  // the first Det. It works ... for a start. Later on this should
  // be taken from the universal XML description of the detector

  return theDets.front()->globalRotation();

}


//__________________________________________________________________________________________________
std::ostream &operator << ( std::ostream &os, const AlignableTrackerPetal & r )
{

  os << "     This Petal contains " << r.theDets.size() << " units" << std::endl;
  os << "     position " << r.globalPosition() << std::endl;
  os << "    (phi,r,z) =  (" << r.globalPosition().phi() << "," 
     << r.globalPosition().perp() << " , " << r.globalPosition().z();
  os << "), orientation:" << std::endl<< r.globalRotation() << std::endl;
  os << r.rotation() << std::endl;

  for ( std::vector<AlignableDet*>::const_iterator idet = r.theDets.begin(); 
		idet != r.theDets.end(); idet++) 
	{
	  for (int i=0;i<(*idet)->size(); i++) 
		{
		  os << "      Det" << i << " position, phi, r: " 
			 << (*idet)->detUnit(i).globalPosition() << ", "
			 << (*idet)->detUnit(i).globalPosition().phi() << ", "
			 << (*idet)->detUnit(i).globalPosition().perp() << std::endl; 
		  
		  os << "      local position, phi, r: " 
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() ) << ", "
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() ).phi() << ", "
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() ).perp() << std::endl; 
		}
	}
  return os;

}











