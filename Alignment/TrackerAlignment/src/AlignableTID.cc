#include "Alignment/TrackerAlignment/interface/AlignableTID.h"


/// The constructor simply copies the vector of layers and computes the surface from them
AlignableTID::AlignableTID( const std::vector<AlignableTIDLayer*> tidLayers )  
{

  theLayers.insert( theLayers.end(), tidLayers.begin(), tidLayers.end() );

  setSurface(computeSurface());

}


/// Clean delete of the vector and its elements
AlignableTID::~AlignableTID() 
{

  for ( std::vector<AlignableTIDLayer*>::iterator iter = theLayers.begin(); 
		iter != theLayers.end(); iter++)
    delete *iter;
  
}


/// Return AlignableLayer at given index
AlignableTIDLayer &AlignableTID::layer(int i) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *theLayers[i];
  
}


/// Returns surface corresponding to current position
/// and orientation, as given by average on all components
AlignableSurface AlignableTID::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );
  
}


/// Compute average z position from all components (x and y forced to 0)
AlignableTID::PositionType AlignableTID::computePosition() 
{

  float zz=0.;
  
  //maybe there exists an easier solution via the layeraccessor class
  // put I was not successfull for the time being
  
  for ( std::vector<AlignableTIDLayer*>::iterator ilayer=theLayers.begin();
		ilayer != theLayers.end(); ilayer++ )
	zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theLayers.size());

  return PositionType( 0.0, 0.0, zz );

}


/// Just initialize to default given by default constructor of a RotationType
AlignableTID::RotationType AlignableTID::computeOrientation() 
{

  return RotationType();

}


/// Output TID information
std::ostream &operator << (std::ostream& os, const AlignableTID& b )
{

  os << "This TID contains " << b.theLayers.size() << " TID layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


/// Recursive printout of whole TID structure
void AlignableTID::dump( void )
{

  // Print the whole structure
  
  std::cout << (*this);
  for ( std::vector<AlignableTIDLayer*>::iterator iLayer = theLayers.begin();
		iLayer != theLayers.end(); iLayer++ )
	(*iLayer)->dump();
	

}



