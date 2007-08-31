/**
 * \file PedeLabeler.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.14 $
 *  $Date: 2007/07/12 17:32:39 $
 *  (last update by $Author: flucke $)
 */

#include "PedeLabeler.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"


// NOTE: Changing '+4' makes older binary files unreadable...
const unsigned int PedeLabeler::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM + 4;
const unsigned int PedeLabeler::theMinLabel = 1; // must be > 0

//___________________________________________________________________________
PedeLabeler::PedeLabeler(Alignable *ali1, Alignable *ali2)
{
  std::vector<Alignable*> alis;
  alis.push_back(ali1);
  alis.push_back(ali2);

  this->buildMap(alis);
}

//___________________________________________________________________________
PedeLabeler::PedeLabeler(const std::vector<Alignable*> &alis)
{
  this->buildMap(alis);
}

//___________________________________________________________________________

PedeLabeler::~PedeLabeler()
{
}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int PedeLabeler::alignableLabel(Alignable *alignable) const
{
  if (!alignable) return 0;

  AlignableToIdMap::const_iterator position = myAlignableToIdMap.find(alignable);
  if (position != myAlignableToIdMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError")
      << "@SUB=PedeLabeler::alignableLabel" << "Alignable "
      << typeid(*alignable).name() << " not in map";
    return 0;
  }

  /*
// following ansatz does not work since the maximum label allowed by pede is 99 999 999...
//   TrackerAlignableId idProducer;
//   const DetId detId(alignable->geomDetId()); // does not work: only AlignableDet(Unit) has DetId...
//  if (detId.det() != DetId::Tracker) {
  const unsigned int detOffset = 28; // would like to use definition from DetId
  const TrackerAlignableId::UniqueId uniqueId(idProducer.alignableUniqueId(alignable));
  const uint32_t detId = uniqueId.first; // uniqueId is a pair...
  const uint32_t det = detId >> detOffset; // most significant bits are detector part
  if (det != DetId::Tracker) {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError") << "@SUB=PedeLabeler::alignableLabel "
      << "Expecting DetId::Tracker (=" << DetId::Tracker << "), but found "
      << det << " which would make the pede labels ambigous. "
      << typeid(*alignable).name() << " " << detId;
    return 0;
  }
  // FIXME: Want: const AlignableObjectId::AlignableObjectIdType type = 
  const unsigned int aType = static_cast<unsigned int>(uniqueId.second);// alignable->alignableObjectId();
  if (aType != ((aType << detOffset) >> detOffset)) {
    // i.e. too many bits (luckily we are  not the muon system...)
    throw cms::Exception("LogicError")  << "@SUB=PedeLabeler::alignableLabel "
      << "Expecting alignableTypeId with at most " << 32 - detOffset
      << " bits, but the number is " << aType
      << " which would make the pede labels ambigous.";
    return 0;
  }

  const uint32_t detIdWithoutDet = (detId - (det << detOffset));
  return detIdWithoutDet + (aType << detOffset);
*/
}

//_________________________________________________________________________
unsigned int PedeLabeler::parameterLabel(unsigned int aliLabel, unsigned int parNum) const
{
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=PedeLabeler::parameterLabel" 
                                      << "Parameter number " << parNum 
                                      << " out of range 0 <= num < " << theMaxNumParam;
  }
  return aliLabel + parNum;
}

//___________________________________________________________________________
unsigned int PedeLabeler::paramNumFromLabel(unsigned int paramLabel) const
{
  if (paramLabel < theMinLabel) {
    edm::LogError("LogicError") << "@SUB=PedeLabeler::paramNumFromLabel"
                                << "Label " << paramLabel << " should be >= " << theMinLabel;
    return 0;
  }
  return (paramLabel - theMinLabel) % theMaxNumParam;
}

//___________________________________________________________________________
unsigned int PedeLabeler::alignableLabelFromLabel(unsigned int paramLabel) const
{
  return paramLabel - this->paramNumFromLabel(paramLabel);
}

//___________________________________________________________________________
Alignable* PedeLabeler::alignableFromLabel(unsigned int label) const
{
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel) return 0; // error already given
  
  if (myIdToAlignableMap.empty()) const_cast<PedeLabeler*>(this)->buildReverseMap();
  IdToAlignableMap::const_iterator position = myIdToAlignableMap.find(aliLabel);
  if (position != myIdToAlignableMap.end()) {
    return position->second;
  } else {
    edm::LogError("LogicError") << "@SUB=PedeLabeler::alignableFromLabel"
                                << "Alignable label " << aliLabel << " not in map";
    return 0;
  }
}

// //_________________________________________________________________________
// bool PedeLabeler::isNoHiera(const Alignable* ali) const
// {
//   return (myNoHieraCollection.find(ali) != myNoHieraCollection.end());
// }

//_________________________________________________________________________
unsigned int PedeLabeler::buildMap(const std::vector<Alignable*> &alis)
{

  myAlignableToIdMap.clear(); // just in case of re-use...

  std::vector<Alignable*> allComps;

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(); iAli != alis.end(); ++iAli) {
    if (*iAli) {
      allComps.push_back(*iAli);
      (*iAli)->recursiveComponents(allComps);
    }
  }

  unsigned int id = theMinLabel;
  for (std::vector<Alignable*>::const_iterator iter = allComps.begin();
       iter != allComps.end(); ++iter) {
    myAlignableToIdMap.insert(AlignableToIdPair(*iter, id));
    id += theMaxNumParam;
  }
  
  return allComps.size();
}


//_________________________________________________________________________
unsigned int PedeLabeler::buildReverseMap()
{

  myIdToAlignableMap.clear();  // just in case of re-use...

  for (AlignableToIdMap::iterator it = myAlignableToIdMap.begin();
       it != myAlignableToIdMap.end(); ++it) {
    const unsigned int key = (*it).second;
    Alignable *ali = (*it).first;
    myIdToAlignableMap[key] = ali;
  }

  return myIdToAlignableMap.size();
}

/*
//_________________________________________________________________________
unsigned int PedeLabeler::buildNoHierarchyCollection(const std::vector<Alignable*> &alis)
{
  myNoHieraCollection.clear();  // just in case of re-use...

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;
    // Now check whether taking out of hierarchy is selected - must be consistent!
    unsigned int numNoHieraPar = 0;
    unsigned int numHieraPar = 0;
    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      const char selector = selVar->fullSelection()[iParam];
      if (selector == 'C' || selector == 'F' || selector == 'H') {
	++numNoHieraPar;
      } else if (selector == 'c' || selector == 'f' || selector == '1' || selector == 'r') {
	++numHieraPar;
      } // else ... accept '0' as undetermined
    }
    if (numNoHieraPar) { // Selected to be taken out.
      if (numHieraPar) { // Inconsistent: Some parameters still in hierarchy ==> exception!
	throw cms::Exception("BadConfig") 
	  << "[PedeLabeler::buildNoHierarchyCollection] All active parameters of alignables to be "
	  << " taken out of the hierarchy must be marked with capital letters 'C', 'F' or 'H'!";
      }
      bool isInHiera = false; // Check whether Alignable is really part of hierarchy:
      Alignable *mother = *iAli;
      while ((mother = mother->mother())) {
	if (mother->alignmentParameters()) isInHiera = true; // could 'break;', but loop is short
      }
      // Complain, but keep collection short if not in hierarchy:
      if (isInHiera) myNoHieraCollection.insert(*iAli);
      else edm::LogWarning("Alignment") << "@SUB=PedeLabeler::buildNoHierarchyCollection"
					<< "Alignable not in hierarchy, no need to remove it!";
    }
  } // end loop on alignables

  return myNoHieraCollection.size();
}


*/
