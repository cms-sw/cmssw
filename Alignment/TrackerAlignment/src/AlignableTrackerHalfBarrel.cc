#include "Alignment/TrackerAlignment/interface/AlignableTrackerHalfBarrel.h"


//__________________________________________________________________________________________________
AlignableTrackerHalfBarrel::AlignableTrackerHalfBarrel
( const std::vector<AlignableTrackerBarrelLayer*> barrelLayers ) 
{

  theBarrelLayers.insert( theBarrelLayers.end(), barrelLayers.begin(), barrelLayers.end() );

  setSurface( computeSurface() );
   
}
      

//__________________________________________________________________________________________________
AlignableTrackerHalfBarrel::~AlignableTrackerHalfBarrel() 
{
  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iter = theBarrelLayers.begin(); 
	iter != theBarrelLayers.end(); iter++)
    delete *iter;

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableTrackerHalfBarrel::components() const
{
  
  std::vector<Alignable*> result; 
  result.insert( result.end(), theBarrelLayers.begin(), theBarrelLayers.end() );
  return result;
  
}


//__________________________________________________________________________________________________
AlignableTrackerBarrelLayer &AlignableTrackerHalfBarrel::layer(int i) 
{
  
  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *theBarrelLayers[i];
  
}


//__________________________________________________________________________________________________
void AlignableTrackerHalfBarrel::twist(float rad) 
{

  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iter = theBarrelLayers.begin();
		iter != theBarrelLayers.end(); iter++ ) 
	(*iter)->twist(rad);
  
}



//__________________________________________________________________________________________________
AlignableSurface AlignableTrackerHalfBarrel::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//__________________________________________________________________________________________________
AlignableTrackerHalfBarrel::PositionType AlignableTrackerHalfBarrel::computePosition() 
{

  float zz = 0.;

  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator ilayer = theBarrelLayers.begin();
		ilayer != theBarrelLayers.end(); ilayer++ )
    zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>(theBarrelLayers.size());

  return PositionType( 0.0, 0.0, zz );

}


//__________________________________________________________________________________________________
AlignableTrackerHalfBarrel::RotationType AlignableTrackerHalfBarrel::computeOrientation() 
{
  return RotationType();
}


//__________________________________________________________________________________________________
std::ostream &operator << (std::ostream& os, const AlignableTrackerHalfBarrel& b )
{

  os << "This HalfBarrel contains " << b.theBarrelLayers.size() << " Barrel layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//__________________________________________________________________________________________________
void AlignableTrackerHalfBarrel::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignableTrackerBarrelLayer*>::iterator iLayer = theBarrelLayers.begin();
		iLayer != theBarrelLayers.end(); iLayer++ )
	(*iLayer)->dump();

}
