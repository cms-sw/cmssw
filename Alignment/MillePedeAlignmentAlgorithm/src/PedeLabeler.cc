/**
 * \file PedeLabeler.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.4 $
 *  $Date: 2008/07/30 15:42:04 $
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
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError") 
    edm::LogError("LogicError")
      << "@SUB=PedeLabeler::alignableLabel" << "Alignable "
      << typeid(*alignable).name() << " not in map, det/subdet/alignableStructureType = "
      << detId.det() << "/" << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//_________________________________________________________________________
unsigned int PedeLabeler::lasBeamLabel(unsigned int lasBeamId) const
{
  UintUintMap::const_iterator position = myLasBeamToLabelMap.find(lasBeamId);
  if (position != myLasBeamToLabelMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogError("LogicError") << "@SUB=PedeLabeler::lasBeamLabel"
				<< "No label for beam Id " << lasBeamId;
    return 0;
  }
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
    // error only if not in lasBeamMap:
    UintUintMap::const_iterator position = myLabelToLasBeamMap.find(aliLabel);
    if (position == myLabelToLasBeamMap.end()) {
      edm::LogError("LogicError") << "@SUB=PedeLabeler::alignableFromLabel"
				  << "Alignable label " << aliLabel << " not in map.";
    }
    return 0;
  }
}

//___________________________________________________________________________
unsigned int PedeLabeler::lasBeamIdFromLabel(unsigned int label) const
{
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel) return 0; // error already given
  
  if (myLabelToLasBeamMap.empty()) const_cast<PedeLabeler*>(this)->buildReverseMap();
  UintUintMap::const_iterator position = myLabelToLasBeamMap.find(aliLabel);
  if (position != myLabelToLasBeamMap.end()) {
    return position->second;
  } else {
    edm::LogError("LogicError") << "@SUB=PedeLabeler::lasBeamIdFromLabel"
				<< "Alignable label " << aliLabel << " not in map.";
    return 0;
  }
}

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
  
  // also care about las beams
  myLasBeamToLabelMap.clear(); // just in case of re-use...
  // FIXME: Temporarily hard code values stolen from 
  // https://twiki.cern.ch/twiki/bin/view/CMS/TkLasTrackBasedInterface#Beam_identifier .
  unsigned int beamIds[] = {  0,  10,  20,  30,  40,  50,  60,  70, // TEC+ R4
			      1,  11,  21,  31,  41,  51,  61,  71, // TEC+ R6
			    100, 110, 120, 130, 140, 150, 160, 170, // TEC- R4
			    101, 111, 121, 131, 141, 151, 161, 171, // TEC- R6
			    200, 210, 220, 230, 240, 250, 260, 270};// AT

  const size_t nBeams = sizeof(beamIds)/sizeof(beamIds[0]);
  for (size_t iBeam = 0; iBeam < nBeams; ++iBeam) {
    //edm::LogInfo("Alignment") << "Las beam " << beamIds[iBeam] << " gets label " << id << ".";
    myLasBeamToLabelMap[beamIds[iBeam]] = id;
    id += theMaxNumParam;
  }

  // return combined size
  return myAlignableToIdMap.size() + myLasBeamToLabelMap.size();
}

//_________________________________________________________________________
unsigned int PedeLabeler::buildReverseMap()
{

  // alignables
  myIdToAlignableMap.clear();  // just in case of re-use...

  for (AlignableToIdMap::iterator it = myAlignableToIdMap.begin();
       it != myAlignableToIdMap.end(); ++it) {
    const unsigned int key = (*it).second;
    Alignable *ali = (*it).first;
    myIdToAlignableMap[key] = ali;
  }

  // las beams
  myLabelToLasBeamMap.clear(); // just in case of re-use...

  for (UintUintMap::const_iterator it = myLasBeamToLabelMap.begin();
       it != myLasBeamToLabelMap.end(); ++it) {
    myLabelToLasBeamMap[it->second] = it->first; //revert key/value
  }

  // return combined size
  return myIdToAlignableMap.size() + myLabelToLasBeamMap.size();
}

