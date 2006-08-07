#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

//__________________________________________________________________________________________________
AlignableNavigator::AlignableNavigator( Alignable* alignable )
{

  theMap.clear();

  recursiveGetId( alignable );


}

//__________________________________________________________________________________________________
AlignableNavigator::AlignableNavigator( std::vector<Alignable*> alignables )
{

  theMap.clear();
 
  
  for ( std::vector<Alignable*>::iterator it = alignables.begin();
		it != alignables.end(); it++ )
	recursiveGetId( *it );


}


//__________________________________________________________________________________________________
Alignable* AlignableNavigator::alignableFromGeomDet( const GeomDet* geomDet )
{
   return alignableFromDetId( geomDet->geographicalId() );
}


//__________________________________________________________________________________________________
Alignable* AlignableNavigator::alignableFromDetId( const DetId& detid )
{

  MapType::iterator position = theMap.find( detid );
  if ( position != theMap.end() ) return position->second;

  throw cms::Exception("BadLogic") << "DetId " << detid.rawId() << " not found";

}



//__________________________________________________________________________________________________
void AlignableNavigator::recursiveGetId( Alignable* alignable )
{
  
  // Recursive method to get the detIds of an alignable and its childs
  // and add the to the map

  if ( alignable->geomDetId().rawId() ) 
	theMap.insert( PairType( alignable->geomDetId(), alignable ) );
  for ( std::vector<Alignable*>::iterator it = alignable->components().begin();
		it != alignable->components().end(); it++ )
	this->recursiveGetId( *it );

}
