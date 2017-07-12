//  \file AlignableNavigator.cc
//
//   $Revision: 1.22 $
//   $Date: 2010/09/10 10:30:03 $
//   (last update by $Author: mussgill $)

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

//_____________________________________________________________________________
AlignableNavigator::AlignableNavigator(Alignable* tracker, Alignable* muon)
{
  theMap.clear();

  const unsigned int numNonDets = this->recursiveGetId(tracker) + this->recursiveGetId(muon);

  if (numNonDets) {
    edm::LogWarning("Alignment") <<"@SUB=AlignableNavigator" << "Created with map of size "
                                 << theMap.size() << ", but found also " << numNonDets 
                                 << " Alignables that have DetId!=0,\nbeing neither "
				 << "AlignableDet nor AlignableDetUnit. This will "
                                 << "lead to an exception in case alignableFromDetId(..) "
				 << "is called for one of these DetIds.\n" 
                                 << "If there is no exception, you can ignore this message.";
  } else {
    edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "Created with map of size "
                              << theMap.size() << ".";
  }
}

//_____________________________________________________________________________
AlignableNavigator::AlignableNavigator(AlignableExtras* extras, Alignable* tracker, Alignable* muon)
{
  theMap.clear();

  unsigned int numNonDets = this->recursiveGetId(tracker) + this->recursiveGetId(muon);

  if (extras) {
    align::Alignables allExtras = extras->components();
    for (auto & allExtra : allExtras) {
      numNonDets += this->recursiveGetId(allExtra);
    }
  }

  if (numNonDets) {
    edm::LogWarning("Alignment") <<"@SUB=AlignableNavigator" << "Created with map of size "
                                 << theMap.size() << ", but found also " << numNonDets 
                                 << " Alignables that have DetId!=0,\nbeing neither "
				 << "AlignableDet nor AlignableDetUnit. This will "
                                 << "lead to an exception in case alignableFromDetId(..) "
				 << "is called for one of these DetIds.\n" 
                                 << "If there is no exception, you can ignore this message.";
  } else {
    edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "Created with map of size "
                              << theMap.size() << ".";
  }
}

//_____________________________________________________________________________
AlignableNavigator::AlignableNavigator( const std::vector<Alignable*>& alignables )
{
  theMap.clear();

  unsigned int numNonDets = 0;
  for (auto alignable : alignables) {
    numNonDets += this->recursiveGetId(alignable);
  }
  if (numNonDets) {
    edm::LogWarning("Alignment") <<"@SUB=AlignableNavigator" << "Created with map of size "
                                 << theMap.size() << ", but found also " << numNonDets 
                                 << " Alignables that have DetId!=0,\nbeing neither "
				 << "AlignableDet nor AlignableDetUnit. This will "
                                 << "lead to an exception in case alignableFromDetId(..) "
				 << "is called for one of these DetIds.\n" 
                                 << "If there is no exception, you can ignore this message.";
  } else {
    edm::LogInfo("Alignment") <<"@SUB=AlignableNavigator" << "created with map of size "
                              << theMap.size() << ".";
  }
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
  if ( position != theMap.end() ) {
    return position->second;
  }
   
  throw cms::Exception("BadLogic") 
    << "[AlignableNavigator::alignableDetFromDetId] DetId " << detid.rawId() << " not found";

  return static_cast<AlignableDet*>(0);
}

//_____________________________________________________________________________
unsigned int AlignableNavigator::recursiveGetId( Alignable* alignable )
{
  // Recursive method to get the detIds of an alignable and its childs
  // and add them to the map.
  // Returns number of Alignables with DetId which are neither AlignableDet
  // nor AlignableDetUnit and are thus not added to the map.

  if (!alignable) return 0;

  unsigned int nProblem = 0;
  const DetId detId(alignable->geomDetId());
  if (detId.rawId()) {
    
    AlignableDet *aliDet;
    AlignableDetUnit *aliDetUnit;
    AlignableBeamSpot *aliBeamSpot;

    if ((aliDet = dynamic_cast<AlignableDet*>(alignable))) {
      
      theMap.insert( PairType( detId, aliDet ) );
    
    } else if ((aliDetUnit = dynamic_cast<AlignableDetUnit*>(alignable))) {

      theMap.insert( PairType( detId, aliDetUnit ) );

    } else if ((aliBeamSpot = dynamic_cast<AlignableBeamSpot*>(alignable))) {

      theMap.insert( PairType( detId, aliBeamSpot ) );

    } else {
      
      nProblem = 1; // equivalent to '++nProblem;' which could confuse to be ina loop...
      // Cannot be an exception since it happens (illegaly) in Muon DT hierarchy:
      //         throw cms::Exception("BadLogic") 
      //           << "[AlignableNavigator::recursiveGetId] Alignable with DetId " << detId.rawId() 
      //           << " neither AlignableDet nor AlignableDetUnit";
    }
    
    if (!nProblem && !this->detAndSubdetInMap(detId)) {
      theDetAndSubdet.push_back(std::pair<int, int>( detId.det(), detId.subdetId() ));
    }
  }
  std::vector<Alignable*> comp = alignable->components();
  if ( alignable->alignableObjectId() != align::AlignableDet
       || comp.size() > 1 ) { // Non-glued AlignableDets contain themselves
    for (auto & it : comp) {
      nProblem += this->recursiveGetId(it);
    }
  }
  return nProblem;
}

//_____________________________________________________________________________
std::vector<AlignableDetOrUnitPtr>
AlignableNavigator::alignablesFromHits( const std::vector<const TransientTrackingRecHit*>& hitvec )
{
  std::vector<AlignableDetOrUnitPtr> result;
  result.reserve(hitvec.size());

  for(auto ih : hitvec) {
    result.push_back(this->alignableFromDetId(ih->geographicalId()));
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
  
  for (const auto & it : hitVec) {
    result.push_back(this->alignableFromDetId(it->geographicalId()));
  }

  return result;
}

//_____________________________________________________________________________
std::vector<AlignableDetOrUnitPtr> AlignableNavigator::alignableDetOrUnits()
{
  std::vector<AlignableDetOrUnitPtr> result;
  result.reserve(theMap.size());

  for (MapType::const_iterator iIdAli = theMap.begin(); iIdAli != theMap.end(); ++iIdAli) {
    result.push_back(iIdAli->second);
  }

  return result;
}

//_____________________________________________________________________________
bool AlignableNavigator::detAndSubdetInMap( const DetId& detid ) const
{
   int det = detid.det();
   int subdet = detid.subdetId();
   for (const auto & i : theDetAndSubdet) {
      if (det == i.first  &&  subdet == i.second) return true;
   }
   return false;
}
