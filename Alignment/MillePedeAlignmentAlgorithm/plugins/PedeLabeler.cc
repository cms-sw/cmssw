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
PedeLabeler::PedeLabeler(const PedeLabelerBase::TopLevelAlignables& alignables, const edm::ParameterSet& config)
    : PedeLabelerBase(alignables, config) {
  align::Alignables alis;
  alis.push_back(alignables.aliTracker_);
  alis.push_back(alignables.aliMuon_);

  if (alignables.aliExtras_) {
    for (const auto& ali : alignables.aliExtras_->components()) {
      alis.push_back(ali);
    }
  }

  this->buildMap(alis);
  this->buildReverseMap();
}

//___________________________________________________________________________
PedeLabeler::~PedeLabeler() {}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int PedeLabeler::alignableLabel(const Alignable* alignable) const {
  if (!alignable)
    return 0;

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    return position->second;
  } else {
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError")
    edm::LogError("LogicError") << "@SUB=PedeLabeler::alignableLabel"
                                << "Alignable " << typeid(*alignable).name()
                                << " not in map, det/subdet/alignableStructureType = " << detId.det() << "/"
                                << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//___________________________________________________________________________
// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int PedeLabeler::alignableLabelFromParamAndInstance(const Alignable* alignable,
                                                             unsigned int /*param*/,
                                                             unsigned int /*instance*/) const {
  return this->alignableLabel(alignable);
}

//_________________________________________________________________________
unsigned int PedeLabeler::lasBeamLabel(unsigned int lasBeamId) const {
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
unsigned int PedeLabeler::parameterLabel(unsigned int aliLabel, unsigned int parNum) const {
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=PedeLabeler::parameterLabel"
                                      << "Parameter number " << parNum << " out of range 0 <= num < " << theMaxNumParam;
  }
  return aliLabel + parNum;
}

//___________________________________________________________________________
unsigned int PedeLabeler::paramNumFromLabel(unsigned int paramLabel) const {
  if (paramLabel < theMinLabel) {
    edm::LogError("LogicError") << "@SUB=PedeLabeler::paramNumFromLabel"
                                << "Label " << paramLabel << " should be >= " << theMinLabel;
    return 0;
  }
  return (paramLabel - theMinLabel) % theMaxNumParam;
}

//___________________________________________________________________________
unsigned int PedeLabeler::alignableLabelFromLabel(unsigned int paramLabel) const {
  return paramLabel - this->paramNumFromLabel(paramLabel);
}

//___________________________________________________________________________
Alignable* PedeLabeler::alignableFromLabel(unsigned int label) const {
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel)
    return nullptr;  // error already given

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
    return nullptr;
  }
}

//___________________________________________________________________________
unsigned int PedeLabeler::lasBeamIdFromLabel(unsigned int label) const {
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel)
    return 0;  // error already given

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
unsigned int PedeLabeler::buildMap(const align::Alignables& alis) {
  theAlignableToIdMap.clear();  // just in case of re-use...

  align::Alignables allComps;

  for (const auto& iAli : alis) {
    if (iAli) {
      allComps.push_back(iAli);
      iAli->recursiveComponents(allComps);
    }
  }

  unsigned int id = theMinLabel;
  for (const auto& iter : allComps) {
    theAlignableToIdMap.insert(AlignableToIdPair(iter, id));
    id += theMaxNumParam;
  }

  // also care about las beams
  theLasBeamToLabelMap.clear();  // just in case of re-use...
  // FIXME: Temporarily hard code values stolen from
  // https://twiki.cern.ch/twiki/bin/view/CMS/TkLasTrackBasedInterface#Beam_identifier .
  unsigned int beamIds[] = {0,   10,  20,  30,  40,  50,  60,  70,    // TEC+ R4
                            1,   11,  21,  31,  41,  51,  61,  71,    // TEC+ R6
                            100, 110, 120, 130, 140, 150, 160, 170,   // TEC- R4
                            101, 111, 121, 131, 141, 151, 161, 171,   // TEC- R6
                            200, 210, 220, 230, 240, 250, 260, 270};  // AT

  const size_t nBeams = sizeof(beamIds) / sizeof(beamIds[0]);
  for (size_t iBeam = 0; iBeam < nBeams; ++iBeam) {
    //edm::LogInfo("Alignment") << "Las beam " << beamIds[iBeam] << " gets label " << id << ".";
    theLasBeamToLabelMap[beamIds[iBeam]] = id;
    id += theMaxNumParam;
  }

  // return combined size
  return theAlignableToIdMap.size() + theLasBeamToLabelMap.size();
}

//_________________________________________________________________________
unsigned int PedeLabeler::buildReverseMap() {
  // alignables
  theIdToAlignableMap.clear();  // just in case of re-use...

  for (const auto& it : theAlignableToIdMap) {
    theIdToAlignableMap[it.second] = it.first;
  }

  // las beams
  theLabelToLasBeamMap.clear();  // just in case of re-use...

  for (const auto& it : theLasBeamToLabelMap) {
    theLabelToLasBeamMap[it.second] = it.first;  //revert key/value
  }

  // return combined size
  return theIdToAlignableMap.size() + theLabelToLasBeamMap.size();
}

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"
DEFINE_EDM_PLUGIN(PedeLabelerPluginFactory, PedeLabeler, "PedeLabeler");
