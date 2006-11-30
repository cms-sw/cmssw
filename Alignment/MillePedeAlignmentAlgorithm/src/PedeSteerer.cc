/**
 * \file PedeSteerer.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.5 $
 *  $Date: 2006/11/15 14:26:44 $
 *  (last update by $Author: flucke $)
 */

#include "PedeSteerer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/SelectionUserVariables.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Geometry/Vector/interface/GlobalPoint.h"

#include <fstream>
#include <algorithm>

// from ROOT
#include <TSystem.h>

const unsigned int PedeSteerer::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM;
const unsigned int PedeSteerer::theMinLabel = 1; // must be > 0

//___________________________________________________________________________

PedeSteerer::PedeSteerer(Alignable *highestLevelAlignable, const std::vector<Alignable*> &alis,
//PedeSteerer::PedeSteerer(AlignableTracker *alignableTracker, AlignmentParameterStore *store,
			 const edm::ParameterSet &config) :
  myConfig(config)
{
  // opens steerFileName as text output file, appending .txt

  std::string name(this->directory());
  name += myConfig.getParameter<std::string>("steerFile") += ".txt";
  mySteerFile.open(name.c_str(), std::ios::out);

  if (!mySteerFile.is_open()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::PedeSteerer"
			       << "Could not open " << name << " as output file.";
  }

  this->buildMap(highestLevelAlignable);//alignableTracker);
  const std::pair<unsigned int, unsigned int> nFixFixCor(this->fixParameters(alis));
  edm::LogInfo("Alignment") << "@SUB=PedeSteerer" 
                            << nFixFixCor.first << " parameters fixed at 0. and "
                            << nFixFixCor.second << " at 'original' position";
}

//___________________________________________________________________________

PedeSteerer::~PedeSteerer()
{
  // closes file
  mySteerFile.close();
}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int PedeSteerer::alignableLabel(Alignable *alignable) const
{
  if (!alignable) return 0;

  AlignableToIdMap::const_iterator position = myAlignableToIdMap.find(alignable);
  if (position != myAlignableToIdMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogWarning("LogicError")
      << "@SUB=PedeSteerer::alignableLabel" << "Alignable "
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
                                      << "Parameter number " << parNum 
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
                                << "Label " << paramLabel << " should be >= " << theMinLabel;
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
                                << "Alignable label " << aliLabel << " not in map";
    return 0;
  }
}

//_________________________________________________________________________
float PedeSteerer::cmsToPedeFactor(unsigned int parNum) const
{
  switch (parNum) {
  case RigidBodyAlignmentParameters::dx:
  case RigidBodyAlignmentParameters::dy:
    return 1000.; // cm to mum *1/10 to get smaller values
  case RigidBodyAlignmentParameters::dz:
    return 2500.;   // cm to mum *1/4 
  case RigidBodyAlignmentParameters::dalpha:
  case RigidBodyAlignmentParameters::dbeta:
    return 1000.; // rad to mrad (no first guess for sensitivity yet)
  case RigidBodyAlignmentParameters::dgamma:
    return 10000.; // rad to mrad *10 to get larger values
  default:
    return 1.;
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
std::pair<unsigned int, unsigned int>
PedeSteerer::fixParameters(const std::vector<Alignable*> &alis)
{
  // return number of parameters fixed at 0. and fixed at original position 
  std::pair<unsigned int, unsigned int> numFixNumFixCor(0, 0);

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (!params) continue; // should not happen, but not worth to log an error here...
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;

    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      int whichFix = this->fixParameter(*iAli, iParam, selVar->fullSelection()[iParam]);
      if (whichFix == 1) {
        ++(numFixNumFixCor.first);
      } else if (whichFix == -1) {
        ++(numFixNumFixCor.second);
      }
    }
    params->setUserVariables(0); // erase the info since it is not needed anymore
  }

  // Flush to disc in case we want to use it before closed (e.g. in runPede...), put keep open...
  mySteerFile.flush(); // ...in case this method is called again to add further constraints.

  return numFixNumFixCor;
}

//_________________________________________________________________________
int PedeSteerer::fixParameter(Alignable *ali, unsigned int iParam, char selector)
{
  int result = 0;
  float fixAt = 0.;
  if (selector == 'c') {
    fixAt = RigidBodyAlignmentParameters(ali).parameters()[iParam];//this->origParam(ali, iParam);
    result = -1;
  } else if (selector == 'f') {
    result = 1;
  } else if (selector != '1' && selector != '0') {
    throw cms::Exception("BadConfig")
      << "@SUB=PedeSteerer::fixParameter" << "Unexpected parameter selector '" << selector
      << "', use 'f' (fix), 'c' (fix at correct pos.), '1' (free) or '0' (ignore).";
  }

  if (result) {
    const unsigned int aliLabel = this->alignableLabel(ali);
    mySteerFile << this->parameterLabel(aliLabel, iParam) << "  " << fixAt << " -1.0";
    if (0) { // debug
      const GlobalPoint position(ali->globalPosition());
      mySteerFile << " eta " << position.eta() << ", z " << position.z()
                  << ", r " << position.perp() << ", phi " << position.phi();
    }
    mySteerFile << "\n";
  }

  return result;
}

//_________________________________________________________________________
std::string PedeSteerer::directory() const 
{
  std::string dir(myConfig.getUntrackedParameter<std::string>("fileDir"));
  if (!dir.empty() && dir.find_last_of('/') != dir.size() - 1) {
    dir += '/'; // directory may need '/'
  }

  return dir;
}

//_________________________________________________________________________
std::string PedeSteerer::pedeOutFile() const
{
  return this->directory() += myConfig.getParameter<std::string>("steerFile") += ".log";
}

//_________________________________________________________________________
bool PedeSteerer::runPede(const std::string &binaryFile) const
{
  std::string command(myConfig.getUntrackedParameter<std::string>("pedeCommand"));
  command += "n ";
  command += this->directory() += myConfig.getParameter<std::string>("steerFile");
  command += " ";
  command += binaryFile;
  const std::string dump(myConfig.getUntrackedParameter<std::string>("pedeDump"));
  if (!dump.empty()) {
    command += " > ";
    command += this->directory() += dump;
  }

  edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Start running " << command;
  // FIXME: Recommended interface to system commands?
  int shellReturn = gSystem->Exec(command.c_str());
  edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Pede command returns " << shellReturn;

  return !shellReturn;
}
