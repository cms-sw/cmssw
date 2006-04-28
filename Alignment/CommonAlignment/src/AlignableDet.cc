#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"


//__________________________________________________________________________________________________
AlignableDet::AlignableDet( GeomDet* geomDet ) : AlignableComposite( geomDet )
{
  
  // Behaviour depends on level of components:
  // Check if the AlignableDet is a CompositeDet or a DetUnit
  // In both cases, we have to down-cast these GeomDets to GeomDetUnits
  // and cast away the const...
  
  if ( geomDet->components().size() == 0 ) // Is a DetUnit
	{
	  const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( geomDet );
	  theDetUnits.push_back( 
							new AlignableDetUnit( const_cast<GeomDetUnit*>(tmpGeomDetUnit) ) 
							);
	}
  else // Is a compositeDet: push back all components
	{
	  std::vector< const GeomDet*> geomDets = geomDet->components();
	  for ( std::vector<const GeomDet*>::iterator idet=geomDets.begin(); 
			idet != geomDets.end(); idet++ )
		{
		  
		  const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( *idet );
		  if ( tmpGeomDetUnit ) // Just check down-cast worked...
			theDetUnits.push_back(
								  new AlignableDetUnit( const_cast<GeomDetUnit*>(tmpGeomDetUnit) )
								  );
		}
	}

}


//__________________________________________________________________________________________________
AlignableDet::~AlignableDet() {};


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableDet::components() const 
{

  std::vector<Alignable*> result;
  
  result.insert( result.end(), theDetUnits.begin(), theDetUnits.end() );

  return result;

}


//__________________________________________________________________________________________________
AlignableDetUnit &AlignableDet::geomDetUnit(int i) 
{

  if ( i >= size() ) 
    throw cms::Exception("LogicError") << "DetUnit index (" << i << ") out of range";

  return *theDetUnits[i];

}


//__________________________________________________________________________________________________
void AlignableDet::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  for (std::vector<AlignableDetUnit*>::iterator i=theDetUnits.begin(); 
       i!=theDetUnits.end();i++)
    (*i)->setAlignmentPositionError(ape);

}
