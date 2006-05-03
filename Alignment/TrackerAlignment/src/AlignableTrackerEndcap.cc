#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcap.h"


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcap::AlignableTrackerEndcap( const std::vector<AlignableTrackerEndcapLayer*> endcapLayers )  
{

  theEndcapLayers.insert( theEndcapLayers.end(), endcapLayers.begin(), endcapLayers.end() );

  setSurface( computeSurface() );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcap::~AlignableTrackerEndcap() 
{

  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator iter = theEndcapLayers.begin(); 
		iter != theEndcapLayers.end(); iter++) 
    delete *iter;

}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableTrackerEndcap::components() const
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theEndcapLayers.begin(), theEndcapLayers.end() );
  return result;
  
} 


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcapLayer &AlignableTrackerEndcap::layer(int i) 
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Layer index (" << i << ") out of range";

  return *theEndcapLayers[i];
  
}


//--------------------------------------------------------------------------------------------------
AlignableSurface AlignableTrackerEndcap::computeSurface()
{

  return AlignableSurface( computePosition(), computeOrientation() );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcap::PositionType AlignableTrackerEndcap::computePosition() 
{

  float zz=0.;

  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator ilayer = theEndcapLayers.begin();
	   ilayer != theEndcapLayers.end(); ilayer++ )
	zz += (*ilayer)->globalPosition().z();

  zz /= static_cast<float>( theEndcapLayers.size() );

  return PositionType( 0.0, 0.0, zz );

}


//--------------------------------------------------------------------------------------------------
AlignableTrackerEndcap::RotationType AlignableTrackerEndcap::computeOrientation() 
{

  return RotationType();

}


//--------------------------------------------------------------------------------------------------
std::ostream &operator << (std::ostream& os, const AlignableTrackerEndcap& b )
{

  os << "This Endcap contains " << b.theEndcapLayers.size() << " Endcap layers" << std::endl;
  os << "(phi, r, z) =  (" << b.globalPosition().phi() << "," 
     << b.globalPosition().perp() << "," << b.globalPosition().z();
  os << "),  orientation:" << std::endl<< b.globalRotation() << std::endl;
  return os;

}


//--------------------------------------------------------------------------------------------------
void AlignableTrackerEndcap::dump( void )
{

  // Print the whole structure
  
  std::cout << (*this);
  for ( std::vector<AlignableTrackerEndcapLayer*>::iterator iLayer = theEndcapLayers.begin();
		iLayer != theEndcapLayers.end(); iLayer++ )
	(*iLayer)->dump();

}





