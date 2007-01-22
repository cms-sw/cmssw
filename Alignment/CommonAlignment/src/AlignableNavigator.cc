#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( Alignable* alignable )
{
  theMap.clear();

  recursiveGetId( alignable );

  edm::LogInfo("Alignment") <<"[AlignableNavigator] created with map of size "
                            << theMap.size();
}

//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( Alignable* tracker, Alignable* muon )
{
  theMap.clear();

  recursiveGetId( tracker );
  recursiveGetId( muon );

  edm::LogInfo("Alignment") <<"[AlignableNavigator] created with map of size "
                            << theMap.size();
}


//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( std::vector<Alignable*> alignables )
{
  theMap.clear();

  for ( std::vector<Alignable*>::iterator it = alignables.begin();
	it != alignables.end(); it++ )
    recursiveGetId( *it );

  edm::LogInfo("Alignment") <<"[AlignableNavigator] created with map of size "
    << theMap.size();
}

//_____________________________________________________________________________

Alignable* AlignableNavigator::alignableFromGeomDet( const GeomDet* geomDet )
{
  return alignableFromDetId( geomDet->geographicalId() );
}

//_____________________________________________________________________________

Alignable* AlignableNavigator::alignableFromDetId( const DetId& detid )
{
  MapType::iterator position = theMap.find( detid );
  if ( position != theMap.end() ) return position->second;

  throw cms::Exception("BadLogic") << "[AlignableNavigator::alignableFromDetId] DetId " << detid.rawId() << " not found";
}

//_____________________________________________________________________________

AlignableDet* AlignableNavigator::alignableDetFromGeomDet( const GeomDet* geomDet )
{
  return alignableDetFromDetId( geomDet->geographicalId() );
}

//_____________________________________________________________________________

AlignableDet* AlignableNavigator::alignableDetFromDetId( const DetId& detid )
{
  Alignable* ali = alignableFromDetId(detid);

  AlignableDet* aliDet=dynamic_cast<AlignableDet*>(ali);
  if (!aliDet) {
    Alignable* mother = ali->mother();
    if (mother) aliDet=dynamic_cast<AlignableDet*>(mother);
    else throw cms::Exception("BadLogic") << "[AlignableNavigator::alignableDetFromDetId] Not AlignableDet but also no mother...  ";
  }
  if ( aliDet ) return aliDet;
  else throw cms::Exception("BadAssociation") << "[AlignableNavigator::alignableDetsFromDetId] cannot find AlignableDet associated to DetId!";
}

//_____________________________________________________________________________

void AlignableNavigator::recursiveGetId( Alignable* alignable )
{
  // Recursive method to get the detIds of an alignable and its childs
  // and add the to the map

  if ( alignable->geomDetId().rawId()) 
	theMap.insert( PairType( alignable->geomDetId(), alignable ) );
  std::vector<Alignable*> comp = alignable->components();
  if ( alignable->alignableObjectId() != AlignableObjectId::AlignableDet
       || comp.size() > 1 ) // Non-glued AlignableDets contain themselves
    for ( std::vector<Alignable*>::iterator it = comp.begin(); 
      it != comp.end(); it++ )
      this->recursiveGetId( *it );
}

//_____________________________________________________________________________

std::vector<AlignableDet*> 
AlignableNavigator::alignableDetsFromHits( const std::vector<const TransientTrackingRecHit*>& hitvec )
{
  std::vector<AlignableDet*> alidetvec;

  for(std::vector<const TransientTrackingRecHit*>::const_iterator ih=
    hitvec.begin(); ih!=hitvec.end(); ih++ ) {
    AlignableDet* aliDet = alignableDetFromDetId((*ih)->geographicalId());
    if ( aliDet )
      alidetvec.push_back( aliDet );
    else
      throw cms::Exception("BadAssociation") << "[AlignableNavigator::alignableDetsFromHits] find AlignableDet associated to hit!";
  }

  return alidetvec;

}

//_____________________________________________________________________________

std::vector<AlignableDet*>
AlignableNavigator::alignableDetsFromHits
(const TransientTrackingRecHit::ConstRecHitContainer &hitVec)
{

  std::vector<AlignableDet*> alidetvec;
  alidetvec.reserve(hitVec.size());
  
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator it
         = hitVec.begin(); it != hitVec.end(); ++it) {
    AlignableDet* aliDet = this->alignableDetFromDetId((*it)->geographicalId());
    if (aliDet) alidetvec.push_back(aliDet);
    else 
      throw cms::Exception("BadAssociation") << "@SUB=AlignableNavigator::alignableDetsFromHits "
                                             << "Found no AlignableDet associated to hit!";
  }

  return alidetvec;
}
