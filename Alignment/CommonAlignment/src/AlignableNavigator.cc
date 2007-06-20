//  \file AlignableNavigator.cc
//
//   $Revision: 1.16.2.1 $
//   $Date: 2007/04/30 11:33:57 $
//   (last update by $Author: flucke $)

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( Alignable* alignable )
{
  theMap.clear();

  recursiveGetId( alignable );

  edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "created with map of size "
                            << theMap.size();
}

//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( Alignable* tracker, Alignable* muon )
{
  theMap.clear();

  recursiveGetId( tracker );
  recursiveGetId( muon );

  edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "created with map of size "
                            << theMap.size();
}


//_____________________________________________________________________________

AlignableNavigator::AlignableNavigator( std::vector<Alignable*> alignables )
{
  theMap.clear();

  for ( std::vector<Alignable*>::iterator it = alignables.begin(); it != alignables.end(); ++it )
    recursiveGetId( *it );

  edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "created with map of size "
                            << theMap.size();
}

//_____________________________________________________________________________
AlignableDetOrUnitPtr AlignableNavigator::alignableFromGeomDet( const GeomDet* geomDet )
{
  return alignableFromDetId( geomDet->geographicalId() );
}

//_____________________________________________________________________________
AlignableDetOrUnitPtr AlignableNavigator::alignableFromDetId( const DetId& detid )
{

  MapType::iterator position = theMap.find( detid );
  if ( position != theMap.end() ) return position->second;
  throw cms::Exception("BadLogic") 
    << "[AlignableNavigator::alignableDetFromDetId] DetId " << detid.rawId() << " not found";

  return static_cast<AlignableDet*>(0);
}

//_____________________________________________________________________________
AlignableDet* AlignableNavigator::alignableDetFromGeomDet( const GeomDet* geomDet )
{
  return alignableDetFromDetId( geomDet->geographicalId() );
}

//_____________________________________________________________________________
AlignableDet* AlignableNavigator::alignableDetFromDetId( const DetId &detId )
{
  AlignableDetOrUnitPtr ali = this->alignableFromDetId(detId);
  AlignableDet *aliDet = ali.alignableDet();
  if (!aliDet) {
    AlignableDetUnit *aliDetUnit = ali.alignableDetUnit();
    if (!aliDetUnit) {
      throw cms::Exception("BadAssociation") 
        << "[AlignableNavigator::alignableDetFromDetId]" 
        << " Neither AlignableDet nor AlignableDetUnit";
    }
    aliDet = dynamic_cast<AlignableDet*>(aliDetUnit->mother());
    if (!aliDet) {
      throw cms::Exception("BadLogic") 
        << "[AlignableNavigator::alignableDetFromDetId]" << " AlignableDetUnit, but "
        << (aliDetUnit->mother() ? " mother not an AlignableDet." : "without mother.");
    }
    edm::LogWarning("Alignment") << "@SUB=AlignableNavigator::alignableDetFromDetId"
                                 << "Returning AlignableDet although DetId belongs"
                                 << " to AlignableDetUnit. Might become exception in future,"
                                 << " use alignableFromDetId/GeomDet instead!";
  }
  return aliDet; 
}

//_____________________________________________________________________________

void AlignableNavigator::recursiveGetId( Alignable* alignable )
{
  // Recursive method to get the detIds of an alignable and its childs
  // and add the to the map
  DetId detId(alignable->geomDetId());
  if ( detId.rawId()) {
    AlignableDet *aliDet = dynamic_cast<AlignableDet*>(alignable);
    if (aliDet) {
      theMap.insert( PairType( detId, aliDet ) );
    } else {
      AlignableDetUnit *aliDetUnit = dynamic_cast<AlignableDetUnit*>(alignable);
      if (aliDetUnit) {
        theMap.insert( PairType( detId, aliDetUnit ) );
      } else {
        throw cms::Exception("BadLogic") 
          << "[AlignableNavigator::recursiveGetId] Alignable with DetId " << detId.rawId() 
          << " neither AlignableDet nor AlignableDetUnit";
      }
    }
    if (!this->detAndSubdetInMap( detId )) {
      theDetAndSubdet.push_back(std::pair<int, int>( detId.det(), detId.subdetId() ));
    }
  }
  std::vector<Alignable*> comp = alignable->components();
  if ( alignable->alignableObjectId() != AlignableObjectId::AlignableDet
       || comp.size() > 1 ) { // Non-glued AlignableDets contain themselves
    for ( std::vector<Alignable*>::iterator it = comp.begin(); it != comp.end(); ++it ) {
      this->recursiveGetId( *it );
    }
  }
}

//_____________________________________________________________________________
std::vector<AlignableDetOrUnitPtr>
AlignableNavigator::alignablesFromHits( const std::vector<const TransientTrackingRecHit*>& hitvec )
{
  std::vector<AlignableDetOrUnitPtr> result;
  result.reserve(hitvec.size());

  for(std::vector<const TransientTrackingRecHit*>::const_iterator ih
        = hitvec.begin(), iEnd = hitvec.end(); ih != iEnd; ++ih) {
    result.push_back(this->alignableFromDetId((*ih)->geographicalId()));
  }

  return result;
}

//_____________________________________________________________________________

std::vector<AlignableDetOrUnitPtr>
AlignableNavigator::alignablesFromHits
(const TransientTrackingRecHit::ConstRecHitContainer &hitVec)
{

  std::vector<AlignableDetOrUnitPtr> result;
  result.reserve(hitVec.size());
  
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator it
         = hitVec.begin(), iEnd = hitVec.end(); it != iEnd; ++it) {
    result.push_back(this->alignableFromDetId((*it)->geographicalId()));
  }

  return result;
}

//_____________________________________________________________________________
std::vector<AlignableDet*>
AlignableNavigator::alignableDetsFromHits( const std::vector<const TransientTrackingRecHit*>& hitvec )
{
  std::vector<AlignableDet*> result;
  result.reserve(hitvec.size());

  for(std::vector<const TransientTrackingRecHit*>::const_iterator ih
        = hitvec.begin(), iEnd = hitvec.end(); ih != iEnd; ++ih) {
    result.push_back(this->alignableDetFromDetId((*ih)->geographicalId()));
  }

  return result;
}

//_____________________________________________________________________________
std::vector<AlignableDet*>
AlignableNavigator::alignableDetsFromHits
(const TransientTrackingRecHit::ConstRecHitContainer &hitVec)
{
  std::vector<AlignableDet*> result;
  result.reserve(hitVec.size());
  
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator it
         = hitVec.begin(), iEnd = hitVec.end(); it != iEnd; ++it) {
    result.push_back(this->alignableDetFromDetId((*it)->geographicalId()));
  }

  return result;
}

//_____________________________________________________________________________

bool AlignableNavigator::detAndSubdetInMap( const DetId& detid ) const
{
   int det = detid.det();
   int subdet = detid.subdetId();
   for (std::vector<std::pair<int, int> >::const_iterator i = theDetAndSubdet.begin();  i != theDetAndSubdet.end();  ++i) {
      if (det == i->first  &&  subdet == i->second) return true;
   }
   return false;
}
