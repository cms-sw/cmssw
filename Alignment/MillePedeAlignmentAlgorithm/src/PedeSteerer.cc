/**
 * \file PedeSteerer.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.4 $
 *  $Date: 2006/11/14 08:29:05 $
 *  (last update by $Author: flucke $)
 */

#include "PedeSteerer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include <fstream>
#include <algorithm>

const unsigned int PedeSteerer::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM;
const unsigned int PedeSteerer::theMinLabel = 1; // must be > 0

//___________________________________________________________________________

PedeSteerer::PedeSteerer(AlignableTracker *alignableTracker, AlignmentParameterStore *store,
			 const edm::ParameterSet &config, const char *fileDir) 
{
  // opens steerFileName as text output file
  std::string dir(fileDir);
  if (!dir.empty()) dir += '/';
  mySteerFile.open((dir + config.getParameter<std::string>("steerFile")).c_str(), std::ios::out);

  if (!mySteerFile.is_open()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::PedeSteerer"
			       << "Could not open " << config.getParameter<std::string>("steerFile")
			       << " as output file.";
  }

  this->buildMap(alignableTracker);

  this->fixParameters(store, alignableTracker,
                      config.getParameter<edm::ParameterSet>("fixedParameterSelection"));
}

//___________________________________________________________________________

PedeSteerer::~PedeSteerer()
{
  // closes file
  mySteerFile.close();
}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
/// So far works only within the tracker.
unsigned int PedeSteerer::alignableLabel(Alignable *alignable) const
{
  if (!alignable) return 0;

  AlignableToIdMap::const_iterator position = myAlignableToIdMap.find(alignable);
  if (position != myAlignableToIdMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError")
      << "@SUB=PedeSteerer::alignableLabel" << "alignable "
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
    edm::LogWarning("LogicError") << "@SUB=PedeSteerer::alignableLabel "
      << "Expecting DetId::Tracker (=" << DetId::Tracker << "), but found "
      << det << " which would make the pede labels ambigous. "
      << typeid(*alignable).name() << " " << detId;
    return 0;
  }
  // FIXME: Want: const AlignableObjectId::AlignableObjectIdType type = 
  const unsigned int aType = static_cast<unsigned int>(uniqueId.second);// alignable->alignableObjectId();
  if (aType != ((aType << detOffset) >> detOffset)) {
    // i.e. too many bits (luckily we are  not the muon system...)
    throw cms::Exception("LogicError")  << "@SUB=PedeSteerer::alignableLabel "
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
unsigned int PedeSteerer::parameterLabel(unsigned int aliLabel, unsigned int parNum) const
{
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=PedeSteerer::parameterLabel" 
                                      << "parameter number " << parNum 
                                      << " out of range 0 <= num < " << theMaxNumParam;
  }
  return aliLabel + parNum;

  /*
  const unsigned int bitOffset = 20;
  const unsigned int patterLength = 3;
  unsigned int aMask = 0;
  for (unsigned int i = 0; i < patterLength; ++i) {
    aMask += (1 << i);
  }
  const unsigned int bitMask = (aMask << bitOffset);

  if (aliLabel & bitMask) {
    throw cms::Exception("LogicError") 
      << "bits to put parNum in are not empty. Mask " << bitMask
      << ", aliLabel " << aliLabel;
  }

  if (parNum != ((parNum << bitOffset) >> bitOffset)) {
    throw cms::Exception("LogicError") 
      << "parNum = " << parNum << " requires more than " << patterLength
      << " bits";
  }

  aliLabel += (parNum << bitOffset);
  return aliLabel;
  */
}

//___________________________________________________________________________
unsigned int PedeSteerer::paramNumFromLabel(unsigned int paramLabel) const
{
  if (paramLabel < theMinLabel) {
    edm::LogError("LogicError") << "@SUB=PedeSteerer::paramNumFromLabel"
                                << "label " << paramLabel << " should be >= " << theMinLabel;
    return 0;
  }
  return (paramLabel - theMinLabel) % theMaxNumParam;
}

//___________________________________________________________________________
unsigned int PedeSteerer::alignableLabelFromLabel(unsigned int paramLabel) const
{
  return paramLabel - this->paramNumFromLabel(paramLabel);
}

//___________________________________________________________________________
Alignable* PedeSteerer::alignableFromLabel(unsigned int label) const
{
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel) return 0; // error already given
  
  if (myIdToAlignableMap.empty()) const_cast<PedeSteerer*>(this)->buildReverseMap();
  IdToAlignableMap::const_iterator position = myIdToAlignableMap.find(aliLabel);
  if (position != myIdToAlignableMap.end()) {
    return position->second;
  } else {
    edm::LogError("LogicError") << "@SUB=PedeSteerer::alignableFromLabel"
                                << "alignable label " << aliLabel << " not in map";
    return 0;
  }
}

//_________________________________________________________________________
unsigned int PedeSteerer::buildMap(Alignable *highestLevelAli)
{

  myAlignableToIdMap.clear(); // just in case of re-use...
  if (!highestLevelAli) return 0;

  std::vector<Alignable*> allComps;
  allComps.push_back(highestLevelAli);
  highestLevelAli->recursiveComponents(allComps);

  unsigned int id = theMinLabel;
  for (std::vector<Alignable*>::const_iterator iter = allComps.begin();
       iter != allComps.end(); ++iter) {
    myAlignableToIdMap.insert(AlignableToIdPair(*iter, id));
    id += theMaxNumParam;
  }

  return allComps.size();
}


//_________________________________________________________________________
unsigned int PedeSteerer::buildReverseMap()
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

//_________________________________________________________________________
unsigned int PedeSteerer::fixParameters(AlignmentParameterStore *store,
                                        AlignableTracker *alignableTracker,
					const edm::ParameterSet &config)
{
  // return number of fixed parameters
  if (!store) return 0;

  AlignmentParameterSelector selector(alignableTracker);
  selector.addSelections(config);
  const std::vector<Alignable*> &alignables = selector.selectedAlignables();
  const std::vector<std::vector<bool> > &paramSels = selector.selectedParameters();

  const AlignmentParameterStore::Alignables &storeAli = store->alignables();
  unsigned int numFixed = 0;

  std::vector<Alignable*>::const_iterator iAli = alignables.begin();
  std::vector<std::vector<bool> >::const_iterator iParamSel = paramSels.begin();
  for ( ; iAli != alignables.end() && iParamSel != paramSels.end(); ++iAli, ++iParamSel) {

    const AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (!params) {
      if (find(storeAli.begin(), storeAli.end(), *iAli) != storeAli.end()) {
        edm::LogError("Alignment") << "@SUB=PedeSteerer::fixParameters" 
                                   << "no parameters for Alignable in AlignmentParameterStore";
      } else {
        edm::LogInfo("Alignment") << "@SUB=PedeSteerer::fixParameters" << "ali NOT in store";
      }
      continue;
    }

    const unsigned int aliLabel = this->alignableLabel(*iAli);
    for (int iParam = 0; iParam < params->size(); ++iParam) {
      if (!(*iParamSel)[iParam]) continue;
      ++numFixed;
      mySteerFile << this->parameterLabel(aliLabel, iParam) << "  0.0  -1.0";
      if (0) { // debug
        const GlobalPoint position((*iAli)->globalPosition());
        mySteerFile << " eta " << position.eta() << ", z " << position.z()
                    << ", r " << position.perp() << ", phi " << position.phi();
      }
      mySteerFile << "\n";
    }
  }

  return numFixed;
}
