#include "Alignment/TrackerAlignment/interface/AlignablePixelHalfBarrel.h"


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrel::AlignablePixelHalfBarrel( const std::vector<AlignablePixelHalfBarrelLayer*> 
											  barrelLayers )
{

  thePixelHalfBarrelLayers.insert( thePixelHalfBarrelLayers.end(), 
								barrelLayers.begin(), barrelLayers.end() );

  setSurface( computeSurface() );

}
  

//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrel::~AlignablePixelHalfBarrel() {

  for ( std::vector<AlignablePixelHalfBarrelLayer*>::iterator iter = thePixelHalfBarrelLayers.begin();
		iter != thePixelHalfBarrelLayers.end(); iter++ ) 
    delete *iter;

}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignablePixelHalfBarrel::components() const
{
  
  std::vector<Alignable*> result; 
  result.insert( result.end(), thePixelHalfBarrelLayers.begin(), thePixelHalfBarrelLayers.end() );
  return result;
  
}

//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrelLayer &AlignablePixelHalfBarrel::layer( int i ) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *thePixelHalfBarrelLayers[i];
  
}


//--------------------------------------------------------------------------------------------------
AlignableSurface AlignablePixelHalfBarrel::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrel::PositionType AlignablePixelHalfBarrel::computePosition() 
{

  float xx = 0.;

  for ( std::vector<AlignablePixelHalfBarrelLayer*>::iterator iLayer = thePixelHalfBarrelLayers.begin();
        iLayer != thePixelHalfBarrelLayers.end(); iLayer++ )
    xx += (*iLayer)->globalPosition().x();

  xx /= static_cast<float>( thePixelHalfBarrelLayers.size() );

  return PositionType( xx, 0.0, 0.0 );

}


//--------------------------------------------------------------------------------------------------
AlignablePixelHalfBarrel::RotationType AlignablePixelHalfBarrel::computeOrientation() 
{

  return RotationType();

}


//--------------------------------------------------------------------------------------------------
void AlignablePixelHalfBarrel::twist(float rad) 
{
  
  for ( std::vector<AlignablePixelHalfBarrelLayer*>::iterator iter = thePixelHalfBarrelLayers.begin();
		iter != thePixelHalfBarrelLayers.end(); iter++ ) 
	(*iter)->twist(rad); 
  
}


//--------------------------------------------------------------------------------------------------
std::ostream& operator << (std::ostream& os, const AlignablePixelHalfBarrel& b )
{

  os << "This PixelHalfBarrel contains " << b.thePixelHalfBarrelLayers.size() 
	 << " Barrel layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//--------------------------------------------------------------------------------------------------
void AlignablePixelHalfBarrel::dump( void )
{

  std::cout << (*this);
  for ( std::vector<AlignablePixelHalfBarrelLayer*>::iterator iLayer = thePixelHalfBarrelLayers.begin();
		iLayer != thePixelHalfBarrelLayers.end(); iLayer++ )
	(*iLayer)->dump();
  
}



