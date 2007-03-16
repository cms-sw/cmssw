/**
 * \file PedeSteerer.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.10 $
 *  $Date: 2007/03/16 17:12:31 $
 *  (last update by $Author: flucke $)
 */

#include "PedeSteerer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/SelectionUserVariables.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <fstream>
#include <algorithm>

// from ROOT
#include <TSystem.h>

const unsigned int PedeSteerer::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM;
const unsigned int PedeSteerer::theMinLabel = 1; // must be > 0

//___________________________________________________________________________

PedeSteerer::PedeSteerer(Alignable *highestLevelAlignable, AlignmentParameterStore *store,
			 const edm::ParameterSet &config) :
  myParameterStore(store), myConfig(config)
{

  this->buildMap(highestLevelAlignable); //has to be done first
  const std::vector<Alignable*> &alis = myParameterStore->alignables();  

  const std::string nameFixFile(this->fileName("FixPara"));
  const std::pair<unsigned int, unsigned int> nFixFixCor(this->fixParameters(alis, nameFixFile));
  if (nFixFixCor.first != 0 || nFixFixCor.second != 0) {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer" 
                              << nFixFixCor.first << " parameters fixed at 0. and "
                              << nFixFixCor.second << " at 'original' position, "
                              << "steering file " << nameFixFile << ".";
  } 

  const std::string nameHierarchyFile(this->fileName("Hierarchy"));
  unsigned int nConstraint = this->hierarchyConstraints(alis, nameHierarchyFile);
  if (nConstraint) {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer" 
                              << "Hierarchy constraints for " << nConstraint << " alignables, "
                              << "steering file " << nameHierarchyFile << ".";
  }
}

//___________________________________________________________________________

PedeSteerer::~PedeSteerer()
{
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
double PedeSteerer::cmsToPedeFactor(unsigned int parNum) const
{
  return 1.; // mmh, otherwise would need to FIXME hierarchyConstraint...

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
double PedeSteerer::parameterSign() const
{
  return -1.;
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
PedeSteerer::fixParameters(const std::vector<Alignable*> &alis, const std::string &fileName)
{
  // return number of parameters fixed at 0. and fixed at original position 
  std::pair<unsigned int, unsigned int> numFixNumFixCor(0, 0);

  std::ofstream *filePtr = 0;

  for (std::vector<Alignable*>::const_iterator iAli = alis.begin() ; iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (!params) continue; // should not happen, but not worth to log an error here...
    SelectionUserVariables *selVar = dynamic_cast<SelectionUserVariables*>(params->userVariables());
    if (!selVar) continue;

    for (unsigned int iParam = 0; static_cast<int>(iParam) < params->size(); ++iParam) {
      char selector = selVar->fullSelection()[iParam];
      if (selector == '0' || selector == '1') continue; // free or ignored parameter FIXME: ugly!
      if (!filePtr) {
        filePtr = this->createSteerFile(fileName, true);
        (*filePtr) << "Parameter\n";
      }
      int whichFix = this->fixParameter(*iAli, iParam, selVar->fullSelection()[iParam], *filePtr);
      if (whichFix == 1) {
        ++(numFixNumFixCor.first);
      } else if (whichFix == -1) {
        ++(numFixNumFixCor.second);
      }
    }
    params->setUserVariables(0); // erase the info since it is not needed anymore
  }

  delete filePtr; // automatically flushes, no problem if NULL ptr.   

  return numFixNumFixCor;
}

//_________________________________________________________________________
int PedeSteerer::fixParameter(Alignable *ali, unsigned int iParam, char selector,
                              std::ofstream &file) const
{
  int result = 0;
  float fixAt = 0.;
  if (selector == 'c') {
    fixAt = -this->parameterSign() * RigidBodyAlignmentParameters(ali).parameters()[iParam];
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
    file << this->parameterLabel(aliLabel, iParam) << "  " 
         << fixAt * this->cmsToPedeFactor(iParam) << " -1.0";
    if (0) { // debug
      const GlobalPoint position(ali->globalPosition());
      file << "* eta " << position.eta() << ", z " << position.z()
           << ", r " << position.perp() << ", phi " << position.phi();
    }
    file << "\n";
  }

  return result;
}

//_________________________________________________________________________
unsigned int PedeSteerer::hierarchyConstraints(const std::vector<Alignable*> &alis,
                                               const std::string &fileName)
{
  std::ofstream *filePtr = 0;

  unsigned int nConstraints = 0;
  std::vector<Alignable*> aliDaughts;
  for (std::vector<Alignable*>::const_iterator iA = alis.begin(), iEnd = alis.end();
       iA != iEnd; ++iA) {
    aliDaughts.clear();
    if (!(*iA)->firstCompsWithParams(aliDaughts)) {
      edm::LogError("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
                                 << "Some but not all daughters with params!";
    }
//     edm::LogInfo("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
// 			      << aliDaughts.size() << " ali param components";
    if (aliDaughts.empty()) continue;
//     edm::LogInfo("Alignment") << "@SUB=PedeSteerer::hierarchyConstraints"
// 			      << aliDaughts.size() << " alignable components ("
// 			      << (*iA)->size() << " in total) for " 
// 			      << aliId.alignableTypeName(*iA) 
// 			      << ", layer " << aliId.typeAndLayerFromAlignable(*iA).second
// 			      << ", position " << (*iA)->globalPosition()
// 			      << ", r = " << (*iA)->globalPosition().perp();
    if (!filePtr) filePtr = this->createSteerFile(fileName, true);
    ++nConstraints;
    this->hierarchyConstraint(*iA, aliDaughts, *filePtr);
  }

  delete filePtr; // automatically flushes, no problem if NULL ptr.   

  return nConstraints;
}

//_________________________________________________________________________
void PedeSteerer::hierarchyConstraint(const Alignable *ali,
                                      const std::vector<Alignable*> &components,
                                      std::ofstream &file) const
{
  typedef AlignmentParameterStore::ParameterId ParameterId;

  std::vector<std::vector<ParameterId> > paramIdsVec;
  std::vector<std::vector<float> > factorsVec;
  if (!myParameterStore->hierarchyConstraints(ali, components, paramIdsVec, factorsVec)) {
    edm::LogWarning("Alignment") << "@SUB=PedeSteerer::hierarchyConstraint"
				 << "Problems from store.";
  }

  for (unsigned int iConstr = 0; iConstr < paramIdsVec.size(); ++iConstr) {
    if (true) { //debug
      TrackerAlignableId aliId;
      file << "\n* Nr. " << iConstr << " of " << aliId.alignableTypeName(ali) << " " 
	   << this->alignableLabel(const_cast<Alignable*>(ali)) // ugly cast: FIXME!
	   << ", layer " << aliId.typeAndLayerFromAlignable(ali).second
	   << ", position " << ali->globalPosition()
	   << ", r = " << ali->globalPosition().perp();
    }
    file << "\nConstraint   0.\n"; // in future 'Wconstraint'?
    const std::vector<ParameterId> &parIds = paramIdsVec[iConstr];
    const std::vector<float> &factors = factorsVec[iConstr];
    // parIds.size() == factors.size() granted by myParameterStore->hierarchyConstraints
    for (unsigned int iParam = 0; iParam < parIds.size(); ++iParam) {
      const unsigned int aliLabel = this->alignableLabel(parIds[iParam].first);
      const unsigned int paramLabel = this->parameterLabel(aliLabel, parIds[iParam].second);
      if (true) { // debug
	file << "* for param " << parIds[iParam].second << " of " << aliLabel << "\n";
      }
      // FIXME: multiply by cmsToPedeFactor(subcomponent)/cmsToPedeFactor(mother) (or vice a versa?)
      file << paramLabel << "    " << factors[iParam] << "\n";
    }
  } // end loop on constraints


}

//_________________________________________________________________________
std::ofstream* PedeSteerer::createSteerFile(const std::string &name, bool addToList)
{
  std::ofstream *result = new std::ofstream(name.c_str(), std::ios::out);
  if (!result || !result->is_open()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::createSteerFile"
			       << "Could not open " << name << " as output file.";
    delete result; // in case just open failed
  } else if (addToList) {
    mySteeringFiles.push_back(name); // keep track
  }

  return result;
}


//_________________________________________________________________________
std::string PedeSteerer::fileName(const std::string &addendum) const
{

  std::string name(this->directory());
  name += myConfig.getParameter<std::string>("steerFile");
  name += addendum;
  name += ".txt";

  return name;
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
std::string PedeSteerer::buildMasterSteer(const std::vector<std::string> &binaryFiles)
{
  const std::string nameMasterSteer(this->fileName("Master"));
  std::ofstream *mainSteerPtr = this->createSteerFile(nameMasterSteer, false);
  if (!mainSteerPtr) return "";

  // add steering files to master steering file
  std::ofstream &mainSteerRef = *mainSteerPtr;
  for (unsigned int iFile = 0; iFile < mySteeringFiles.size(); ++iFile) {
    mainSteerRef << mySteeringFiles[iFile] << "\n";
  }

  // add binary files to master steering file
  mainSteerRef << "\nCfiles\n";
  for (unsigned int iFile = 0; iFile < binaryFiles.size(); ++iFile) {
    mainSteerRef << binaryFiles[iFile] << "\n";
  }

  // add method
  mainSteerRef << "\nmethod  " << myConfig.getParameter<std::string>("method") << "\n";

  // add outlier treatment
  const std::vector<std::string> outTr(myConfig.getParameter<std::vector<std::string> >("outlier"));
  mainSteerRef << "\n* Outlier treatment\n";
  for (unsigned int i = 0; i < outTr.size(); ++i) {
    mainSteerRef << outTr[i] << "\n";
  }

  // add further options
  const std::vector<std::string> opt(myConfig.getParameter<std::vector<std::string> >("options"));
  mainSteerRef << "\n* Further options \n";
  for (unsigned int i = 0; i < opt.size(); ++i) {
    mainSteerRef << opt[i] << "\n";
  }

  delete mainSteerPtr;  // close (and flush) again

  return nameMasterSteer;
}

//_________________________________________________________________________
bool PedeSteerer::runPede(const std::string &masterSteer) const
{
  if (masterSteer.empty()) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::runPede" << "Empty master steer file, stop";
    return false;
  }

  std::string command(myConfig.getUntrackedParameter<std::string>("pedeCommand"));
  (command += " ") += masterSteer;
  const std::string dump(myConfig.getUntrackedParameter<std::string>("pedeDump"));
  if (!dump.empty()) {
    command += " > ";
    command += this->directory() += dump;
  }

  edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Start running " << command;
  // FIXME: Recommended interface to system commands?
  int shellReturn = gSystem->Exec(command.c_str());
  if (shellReturn) {
    edm::LogError("Alignment") << "@SUB=PedeSteerer::runPede" << "Command returns " << shellReturn;
  } else {
    edm::LogInfo("Alignment") << "@SUB=PedeSteerer::runPede" << "Command returns " << shellReturn;
  }

  return !shellReturn;
}
