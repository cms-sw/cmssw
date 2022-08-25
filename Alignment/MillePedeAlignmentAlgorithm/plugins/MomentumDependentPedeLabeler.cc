/**
 * \file MomentumDependentPedeLabeler.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.2 $
 *  $Date: 2012/08/10 09:01:11 $
 *  (last update by $Author: flucke $)
 */

#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Parse.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "MomentumDependentPedeLabeler.h"

//___________________________________________________________________________
MomentumDependentPedeLabeler::MomentumDependentPedeLabeler(const PedeLabelerBase::TopLevelAlignables &alignables,
                                                           const edm::ParameterSet &config)
    : PedeLabelerBase(alignables, config),
      theOpenMomentumRange(std::pair<float, float>(0.0, 10000.0)),
      theMaxNumberOfParameterInstances(0) {
  align::Alignables alis;
  alis.push_back(alignables.aliTracker_);
  alis.push_back(alignables.aliMuon_);

  if (alignables.aliExtras_) {
    for (const auto &ali : alignables.aliExtras_->components()) {
      alis.push_back(ali);
    }
  }

  this->buildMomentumDependencyMap(alignables.aliTracker_, alignables.aliMuon_, alignables.aliExtras_, config);
  this->buildMap(alis);
  this->buildReverseMap();  // needed already now to 'fill' theMaxNumberOfParameterInstances
}

//___________________________________________________________________________

MomentumDependentPedeLabeler::~MomentumDependentPedeLabeler() {}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int MomentumDependentPedeLabeler::alignableLabel(const Alignable *alignable) const {
  if (!alignable)
    return 0;

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    return position->second;
  } else {
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError")
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::alignableLabel"
                                << "Alignable " << typeid(*alignable).name()
                                << " not in map, det/subdet/alignableStructureType = " << detId.det() << "/"
                                << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//___________________________________________________________________________
// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int MomentumDependentPedeLabeler::alignableLabelFromParamAndInstance(const Alignable *alignable,
                                                                              unsigned int param,
                                                                              unsigned int instance) const {
  if (!alignable)
    return 0;

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    AlignableToMomentumRangeMap::const_iterator positionAli = theAlignableToMomentumRangeMap.find(alignable);
    if (positionAli != theAlignableToMomentumRangeMap.end()) {
      MomentumRangeParamMap::const_iterator positionParam = (*positionAli).second.find(param);
      if (positionParam != (*positionAli).second.end()) {
        if (instance >= (*positionParam).second.size()) {
          throw cms::Exception("Alignment") << "@SUB=MomentumDependentPedeLabeler::alignableLabelFromParamAndMomentum"
                                            << "iovIdx out of bounds";
        }
        return position->second + instance * theParamInstanceOffset;
      } else {
        return position->second;
      }
    } else {
      return position->second;
    }
  } else {
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError")
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::alignableLabel"
                                << "Alignable " << typeid(*alignable).name()
                                << " not in map, det/subdet/alignableStructureType = " << detId.det() << "/"
                                << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::lasBeamLabel(unsigned int lasBeamId) const {
  UintUintMap::const_iterator position = theLasBeamToLabelMap.find(lasBeamId);
  if (position != theLasBeamToLabelMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError")
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::lasBeamLabel"
                                << "No label for beam Id " << lasBeamId;
    return 0;
  }
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::parameterLabel(unsigned int aliLabel, unsigned int parNum) const {
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=MomentumDependentPedeLabeler::parameterLabel"
                                      << "Parameter number " << parNum << " out of range 0 <= num < " << theMaxNumParam;
  }
  return aliLabel + parNum;
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::parameterLabel(Alignable *alignable,
                                                          unsigned int parNum,
                                                          const AlignmentAlgorithmBase::EventInfo &eventInfo,
                                                          const TrajectoryStateOnSurface &tsos) const {
  if (!alignable)
    return 0;

  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=MomentumDependentPedeLabeler::parameterLabel"
                                      << "Parameter number " << parNum << " out of range 0 <= num < " << theMaxNumParam;
  }

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    AlignableToMomentumRangeMap::const_iterator positionAli = theAlignableToMomentumRangeMap.find(alignable);
    if (positionAli != theAlignableToMomentumRangeMap.end()) {
      MomentumRangeParamMap::const_iterator positionParam = (*positionAli).second.find(parNum);
      if (positionParam != (*positionAli).second.end()) {
        int offset = 0;
        float mom = tsos.globalMomentum().mag();
        const MomentumRangeVector &momentumRanges = (*positionParam).second;
        for (const auto &iMomentum : momentumRanges) {
          if (iMomentum.first <= mom && mom < iMomentum.second) {
            return position->second + offset * theParamInstanceOffset + parNum;
          }
          offset++;
        }
        const DetId detId(alignable->id());
        edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::alignableLabel"
                                    << "Alignable " << typeid(*alignable).name()
                                    << " not in map, det/subdet/alignableStructureType = " << detId.det() << "/"
                                    << detId.subdetId() << "/" << alignable->alignableObjectId();
        return 0;
      } else {
        return position->second + parNum;
      }

    } else {
      return position->second + parNum;
    }
  } else {
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError")
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::alignableLabel"
                                << "Alignable " << typeid(*alignable).name()
                                << " not in map, det/subdet/alignableStructureType = " << detId.det() << "/"
                                << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//_________________________________________________________________________
bool MomentumDependentPedeLabeler::hasSplitParameters(Alignable *alignable) const {
  AlignableToMomentumRangeMap::const_iterator positionAli = theAlignableToMomentumRangeMap.find(alignable);
  if (positionAli != theAlignableToMomentumRangeMap.end())
    return true;
  return false;
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::numberOfParameterInstances(Alignable *alignable, int param) const {
  AlignableToMomentumRangeMap::const_iterator positionAli = theAlignableToMomentumRangeMap.find(alignable);
  if (positionAli != theAlignableToMomentumRangeMap.end()) {
    size_t nMomentums = 1;
    if (param == -1) {
      for (const auto &iParam : (*positionAli).second) {
        nMomentums = std::max(nMomentums, iParam.second.size());
      }
      return nMomentums;
    } else {
      MomentumRangeParamMap::const_iterator iParam = (*positionAli).second.find(param);
      if (iParam != (*positionAli).second.end()) {
        return iParam->second.size();
      } else {
        return 1;
      }
    }
  }

  return 1;
}

//___________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::paramNumFromLabel(unsigned int paramLabel) const {
  if (paramLabel < theMinLabel) {
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::paramNumFromLabel"
                                << "Label " << paramLabel << " should be >= " << theMinLabel;
    return 0;
  }
  return (paramLabel - theMinLabel) % theParamInstanceOffset;
}

//___________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::alignableLabelFromLabel(unsigned int paramLabel) const {
  return paramLabel - this->paramNumFromLabel(paramLabel);
}

//___________________________________________________________________________
Alignable *MomentumDependentPedeLabeler::alignableFromLabel(unsigned int label) const {
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
      edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::alignableFromLabel"
                                  << "Alignable label " << aliLabel << " not in map.";
    }
    return nullptr;
  }
}

//___________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::lasBeamIdFromLabel(unsigned int label) const {
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel)
    return 0;  // error already given

  UintUintMap::const_iterator position = theLabelToLasBeamMap.find(aliLabel);
  if (position != theLabelToLasBeamMap.end()) {
    return position->second;
  } else {
    edm::LogError("LogicError") << "@SUB=MomentumDependentPedeLabeler::lasBeamIdFromLabel"
                                << "Alignable label " << aliLabel << " not in map.";
    return 0;
  }
}

//__________________________________________________________________________________________________
std::vector<std::string> MomentumDependentPedeLabeler::decompose(const std::string &s,
                                                                 std::string::value_type delimiter) const {
  std::vector<std::string> result;

  std::string::size_type previousPos = 0;
  while (true) {
    const std::string::size_type delimiterPos = s.find(delimiter, previousPos);
    if (delimiterPos == std::string::npos) {
      result.push_back(s.substr(previousPos));  // until end
      break;
    }
    result.push_back(s.substr(previousPos, delimiterPos - previousPos));
    previousPos = delimiterPos + 1;  // +1: skip delimiter
  }

  return result;
}

//__________________________________________________________________________________________________
std::vector<unsigned int> MomentumDependentPedeLabeler::convertParamSel(const std::string &selString) const {
  std::vector<unsigned int> result;
  for (std::string::size_type pos = 0; pos < selString.size(); ++pos) {
    if (selString[pos] == '1')
      result.push_back(pos);
  }
  return result;
}

unsigned int MomentumDependentPedeLabeler::buildMomentumDependencyMap(AlignableTracker *aliTracker,
                                                                      AlignableMuon *aliMuon,
                                                                      AlignableExtras *aliExtras,
                                                                      const edm::ParameterSet &config) {
  theAlignableToMomentumRangeMap.clear();

  AlignmentParameterSelector selector(aliTracker, aliMuon, aliExtras);

  std::vector<char> paramSelDumthe(6, '1');

  const auto parameterInstancesVPSet = config.getParameter<std::vector<edm::ParameterSet> >("parameterInstances");

  for (const auto &iter : parameterInstancesVPSet) {
    const auto tempMomentumRanges = iter.getParameter<std::vector<std::string> >("momentumRanges");
    if (tempMomentumRanges.empty()) {
      throw cms::Exception("BadConfig") << "@SUB=MomentumDependentPedeLabeler::buildMomentumDependencyMap\n"
                                        << "MomentumRanges empty\n";
    }

    MomentumRangeVector MomentumRanges;
    float lower;
    float upper;
    for (unsigned int iMomentum = 0; iMomentum < tempMomentumRanges.size(); ++iMomentum) {
      std::vector<std::string> tokens = edm::tokenize(tempMomentumRanges[iMomentum], ":");

      lower = strtod(tokens[0].c_str(), nullptr);
      upper = strtod(tokens[1].c_str(), nullptr);

      MomentumRanges.push_back(std::pair<float, float>(lower, upper));
    }

    const auto selStrings = iter.getParameter<std::vector<std::string> >("selector");
    for (const auto &iSel : selStrings) {
      std::vector<std::string> decompSel(this->decompose(iSel, ','));

      if (decompSel.size() != 2) {
        throw cms::Exception("BadConfig") << "@SUB=MomentumDependentPedeLabeler::buildMomentumDependencyMap\n"
                                          << iSel << " should have at least 2 ','-separated parts\n";
      }

      std::vector<unsigned int> selParam = this->convertParamSel(decompSel[1]);
      selector.clear();
      selector.addSelection(decompSel[0], paramSelDumthe);

      const auto &alis = selector.selectedAlignables();
      for (const auto &iAli : alis) {
        if (iAli->alignmentParameters() == nullptr) {
          throw cms::Exception("BadConfig")
              << "@SUB=MomentumDependentPedeLabeler::buildMomentumDependencyMap\n"
              << "Momentum dependence configured for alignable of type "
              << objectIdProvider().idToString(iAli->alignableObjectId()) << " at (" << iAli->globalPosition().x()
              << "," << iAli->globalPosition().y() << "," << iAli->globalPosition().z() << "), "
              << "but that has no parameters. Please check that all run "
              << "momentum parameters are also selected for alignment.\n";
        }

        for (const auto &iParam : selParam) {
          AlignableToMomentumRangeMap::const_iterator positionAli = theAlignableToMomentumRangeMap.find(iAli);
          if (positionAli != theAlignableToMomentumRangeMap.end()) {
            AlignmentParameters *AliParams = (*positionAli).first->alignmentParameters();
            if (static_cast<int>(selParam[selParam.size() - 1]) >= AliParams->size()) {
              throw cms::Exception("BadConfig") << "@SUB=MomentumDependentPedeLabeler::buildMomentumDependencyMap\n"
                                                << "mismatch in number of parameters\n";
            }

            MomentumRangeParamMap::const_iterator positionParam = (*positionAli).second.find(iParam);
            if (positionParam != (*positionAli).second.end()) {
              throw cms::Exception("BadConfig") << "@SUB=MomentumDependentPedeLabeler::buildMomentumDependencyMap\n"
                                                << "Momentum range for parameter specified twice\n";
            }
          }

          theAlignableToMomentumRangeMap[iAli][iParam] = MomentumRanges;
        }
      }
    }
  }

  return theAlignableToMomentumRangeMap.size();
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::buildMap(const align::Alignables &alis) {
  theAlignableToIdMap.clear();  // just in case of re-use...

  align::Alignables allComps;

  for (const auto &iAli : alis) {
    if (iAli) {
      allComps.push_back(iAli);
      iAli->recursiveComponents(allComps);
    }
  }

  unsigned int id = theMinLabel;
  for (const auto &iter : allComps) {
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

  if (id > theParamInstanceOffset) {  // 'overflow' per instance
    throw cms::Exception("Alignment") << "@SUB=MomentumDependentPedeLabeler::buildMap: "
                                      << "Too many labels per instance (" << id - 1 << ") leading to double use, "
                                      << "increase PedeLabelerBase::theParamInstanceOffset!\n";
  }
  // return combined size
  return theAlignableToIdMap.size() + theLasBeamToLabelMap.size();
}

//_________________________________________________________________________
unsigned int MomentumDependentPedeLabeler::buildReverseMap() {
  // alignables
  theIdToAlignableMap.clear();  // just in case of re-use...

  for (const auto &it : theAlignableToIdMap) {
    const unsigned int key = it.second;
    Alignable *ali = it.first;
    const unsigned int nInstances = this->numberOfParameterInstances(ali, -1);
    theMaxNumberOfParameterInstances = std::max(nInstances, theMaxNumberOfParameterInstances);
    for (unsigned int iInstance = 0; iInstance < nInstances; ++iInstance) {
      theIdToAlignableMap[key + iInstance * theParamInstanceOffset] = ali;
    }
  }

  // las beams
  theLabelToLasBeamMap.clear();  // just in case of re-use...

  for (const auto &it : theLasBeamToLabelMap) {
    theLabelToLasBeamMap[it.second] = it.first;  //revert key/value
  }

  // return combined size
  return theIdToAlignableMap.size() + theLabelToLasBeamMap.size();
}

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"
DEFINE_EDM_PLUGIN(PedeLabelerPluginFactory, MomentumDependentPedeLabeler, "MomentumDependentPedeLabeler");
