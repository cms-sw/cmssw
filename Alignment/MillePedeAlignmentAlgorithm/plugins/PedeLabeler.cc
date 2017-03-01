/**
 * \file PedeLabeler.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.7 $
 *  $Date: 2010/10/26 20:49:42 $
 *  (last update by $Author: flucke $)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "PedeLabeler.h"

//___________________________________________________________________________
PedeLabeler::PedeLabeler(const PedeLabelerBase::TopLevelAlignables& alignables,
			 const edm::ParameterSet& config)
  :PedeLabelerBase(alignables, config)
{
  std::vector<Alignable*> alis;
  alis.push_back(alignables.aliTracker_);
  alis.push_back(alignables.aliMuon_);

  if (alignables.aliExtras_) {
    align::Alignables allExtras = alignables.aliExtras_->components();
    for ( std::vector<Alignable*>::iterator it = allExtras.begin(); it != allExtras.end(); ++it ) {
      alis.push_back(*it);
    }
  }

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

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
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

//___________________________________________________________________________
// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int PedeLabeler::alignableLabelFromParamAndInstance(Alignable *alignable,
							     unsigned int /*param*/,
							     unsigned int /*instance*/) const
{
  return this->alignableLabel(alignable);
}

//_________________________________________________________________________
unsigned int PedeLabeler::lasBeamLabel(unsigned int lasBeamId) const
{
  UintUintMap::const_iterator position = theLasBeamToLabelMap.find(lasBeamId);
  if (position != theLasBeamToLabelMap.end()) {
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
  
  if (theIdToAlignableMap.empty()) const_cast<PedeLabeler*>(this)->buildReverseMap();
  IdToAlignableMap::const_iterator position = theIdToAlignableMap.find(aliLabel);
  if (position != theIdToAlignableMap.end()) {
    return position->second;
  } else {
    // error only if not in lasBeamMap:
    UintUintMap::const_iterator position = theLabelToLasBeamMap.find(aliLabel);
    if (position == theLabelToLasBeamMap.end()) {
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
  
  if (theLabelToLasBeamMap.empty()) const_cast<PedeLabeler*>(this)->buildReverseMap();
  UintUintMap::const_iterator position = theLabelToLasBeamMap.find(aliLabel);
  if (position != theLabelToLasBeamMap.end()) {
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
  theAlignableToIdMap.clear(); // just in case of re-use...

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
    theAlignableToIdMap.insert(AlignableToIdPair(*iter, id));
    id += theMaxNumParam;
  }
  
  // also care about las beams
  theLasBeamToLabelMap.clear(); // just in case of re-use...
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
    theLasBeamToLabelMap[beamIds[iBeam]] = id;
    id += theMaxNumParam;
  }

  // return combined size
  return theAlignableToIdMap.size() + theLasBeamToLabelMap.size();
}

//_________________________________________________________________________
unsigned int PedeLabeler::buildReverseMap()
{

  // alignables
  theIdToAlignableMap.clear();  // just in case of re-use...

  for (AlignableToIdMap::iterator it = theAlignableToIdMap.begin();
       it != theAlignableToIdMap.end(); ++it) {
    const unsigned int key = (*it).second;
    Alignable *ali = (*it).first;
    theIdToAlignableMap[key] = ali;
  }

  // las beams
  theLabelToLasBeamMap.clear(); // just in case of re-use...

  for (UintUintMap::const_iterator it = theLasBeamToLabelMap.begin();
       it != theLasBeamToLabelMap.end(); ++it) {
    theLabelToLasBeamMap[it->second] = it->first; //revert key/value
  }

  // return combined size
  return theIdToAlignableMap.size() + theLabelToLasBeamMap.size();
}

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"
DEFINE_EDM_PLUGIN(PedeLabelerPluginFactory, PedeLabeler, "PedeLabeler");
