/**
 * \file RunRangeDependentPedeLabeler.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.4 $
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

#include "RunRangeDependentPedeLabeler.h"

//___________________________________________________________________________
RunRangeDependentPedeLabeler::RunRangeDependentPedeLabeler(const PedeLabelerBase::TopLevelAlignables& alignables,
							   const edm::ParameterSet& config)
  : PedeLabelerBase(alignables, config),
    theMaxNumberOfParameterInstances(0)
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

  this->buildRunRangeDependencyMap(alignables.aliTracker_,
				   alignables.aliMuon_,
				   alignables.aliExtras_, 
				   config);
  this->buildMap(alis);
  this->buildReverseMap(); // needed already now to 'fill' theMaxNumberOfParameterInstances
}

//___________________________________________________________________________

RunRangeDependentPedeLabeler::~RunRangeDependentPedeLabeler()
{
}

//___________________________________________________________________________
/// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int RunRangeDependentPedeLabeler::alignableLabel(Alignable *alignable) const
{
  if (!alignable) return 0;

  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    return position->second;
  } else {
    const DetId detId(alignable->id());
    //throw cms::Exception("LogicError") 
    edm::LogError("LogicError")
      << "@SUB=RunRangeDependentPedeLabeler::alignableLabel" << "Alignable "
      << typeid(*alignable).name() << " not in map, det/subdet/alignableStructureType = "
      << detId.det() << "/" << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//___________________________________________________________________________
// Return 32-bit unique label for alignable, 0 indicates failure.
unsigned int RunRangeDependentPedeLabeler::alignableLabelFromParamAndInstance(Alignable *alignable,
									      unsigned int param,
									      unsigned int instance) const
{
  if (!alignable) return 0;
  
  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {
    AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(alignable);
    if (positionAli != theAlignableToRunRangeRangeMap.end()) {
      RunRangeParamMap::const_iterator positionParam = (*positionAli).second.find(param);
      if (positionParam!=(*positionAli).second.end()) {
	if (instance>=(*positionParam).second.size()) {
	  throw cms::Exception("Alignment") << "RunRangeDependentPedeLabeler::alignableLabelFromParamAndRunRange: " 
					    << "RunRangeIdx out of bounds.\n";
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
    edm::LogError("LogicError")
      << "@SUB=RunRangeDependentPedeLabeler::alignableLabel" << "Alignable "
      << typeid(*alignable).name() << " not in map, det/subdet/alignableStructureType = "
      << detId.det() << "/" << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  }
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::lasBeamLabel(unsigned int lasBeamId) const
{
  UintUintMap::const_iterator position = theLasBeamToLabelMap.find(lasBeamId);
  if (position != theLasBeamToLabelMap.end()) {
    return position->second;
  } else {
    //throw cms::Exception("LogicError") 
    edm::LogError("LogicError") << "@SUB=RunRangeDependentPedeLabeler::lasBeamLabel"
				<< "No label for beam Id " << lasBeamId;
    return 0;
  }
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::parameterLabel(unsigned int aliLabel, unsigned int parNum) const
{
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=RunRangeDependentPedeLabeler::parameterLabel" 
                                      << "Parameter number " << parNum 
                                      << " out of range 0 <= num < " << theMaxNumParam;
  }
  return aliLabel + parNum;
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::parameterLabel(Alignable *alignable, unsigned int parNum,
							  const AlignmentAlgorithmBase::EventInfo &eventInfo,
							  const TrajectoryStateOnSurface &tsos) const
{
  if (!alignable) return 0;
  
  if (parNum >= theMaxNumParam) {
    throw cms::Exception("Alignment") << "@SUB=RunRangeDependentPedeLabeler::parameterLabel" 
                                      << "Parameter number " << parNum 
                                      << " out of range 0 <= num < " << theMaxNumParam;
  }
  
  AlignableToIdMap::const_iterator position = theAlignableToIdMap.find(alignable);
  if (position != theAlignableToIdMap.end()) {

    AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(alignable);
    if (positionAli != theAlignableToRunRangeRangeMap.end()) {
      
      RunRangeParamMap::const_iterator positionParam = (*positionAli).second.find(parNum);
      if (positionParam!=(*positionAli).second.end()) {
	
	int offset = 0;
 	const RunRangeVector & runRanges = (*positionParam).second;
	for (RunRangeVector::const_iterator iRunRange = runRanges.begin();
	     iRunRange != runRanges.end();
	     ++iRunRange) {
	  if (eventInfo.eventId().run() >= iRunRange->first &&
	      eventInfo.eventId().run() <= iRunRange->second) {
	    return position->second + offset * theParamInstanceOffset + parNum;
	  }
 	  offset++;
 	}
	const DetId detId(alignable->id());
	edm::LogError("LogicError")
	  << "@SUB=RunRangeDependentPedeLabeler::parameterLabel" << "Instance for Alignable "
	  << typeid(*alignable).name() << " not in map, det/subdet/alignableStructureType = "
	  << detId.det() << "/" << detId.subdetId() << "/" << alignable->alignableObjectId()
	  << " for run " << eventInfo.eventId().run();
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
    edm::LogError("LogicError")
      << "@SUB=RunRangeDependentPedeLabeler::parameterLabel" << "Alignable "
      << typeid(*alignable).name() << " not in map, det/subdet/alignableStructureType = "
      << detId.det() << "/" << detId.subdetId() << "/" << alignable->alignableObjectId();
    return 0;
  } 
}

//_________________________________________________________________________
bool RunRangeDependentPedeLabeler::hasSplitParameters(Alignable *alignable) const
{
  AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(alignable);
  if (positionAli != theAlignableToRunRangeRangeMap.end()) return true;
  return false;
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::numberOfParameterInstances(Alignable *alignable, int param) const
{
  AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(alignable);
  if (positionAli != theAlignableToRunRangeRangeMap.end()) {

    size_t nRunRanges = 1;
    if (param==-1) {
      for (RunRangeParamMap::const_iterator iParam = (*positionAli).second.begin();
	   iParam != (*positionAli).second.end();
	   ++iParam) {
	nRunRanges = std::max(nRunRanges, iParam->second.size());
      }
      return nRunRanges;
    } else {
      RunRangeParamMap::const_iterator iParam = (*positionAli).second.find(param);
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
unsigned int RunRangeDependentPedeLabeler::paramNumFromLabel(unsigned int paramLabel) const
{
  if (paramLabel < theMinLabel) {
    edm::LogError("LogicError") << "@SUB=RunRangeDependentPedeLabeler::paramNumFromLabel"
                                << "Label " << paramLabel << " should be >= " << theMinLabel;
    return 0;
  }
  return (paramLabel - theMinLabel) % theMaxNumParam;
}

//___________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::alignableLabelFromLabel(unsigned int paramLabel) const
{
  return paramLabel - this->paramNumFromLabel(paramLabel);
}

//___________________________________________________________________________
Alignable* RunRangeDependentPedeLabeler::alignableFromLabel(unsigned int label) const
{
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel) return 0; // error already given
  
  if (theIdToAlignableMap.empty()) const_cast<RunRangeDependentPedeLabeler*>(this)->buildReverseMap();
  IdToAlignableMap::const_iterator position = theIdToAlignableMap.find(aliLabel);
  if (position != theIdToAlignableMap.end()) {
    return position->second;
  } else {
    // error only if not in lasBeamMap:
    UintUintMap::const_iterator position = theLabelToLasBeamMap.find(aliLabel);
    if (position == theLabelToLasBeamMap.end()) {
      edm::LogError("LogicError") << "@SUB=RunRangeDependentPedeLabeler::alignableFromLabel"
				  << "Alignable label " << aliLabel << " not in map.";
    }
    return 0;
  }
}

//___________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::lasBeamIdFromLabel(unsigned int label) const
{
  const unsigned int aliLabel = this->alignableLabelFromLabel(label);
  if (aliLabel < theMinLabel) return 0; // error already given
  
  if (theLabelToLasBeamMap.empty()) const_cast<RunRangeDependentPedeLabeler*>(this)->buildReverseMap();
  UintUintMap::const_iterator position = theLabelToLasBeamMap.find(aliLabel);
  if (position != theLabelToLasBeamMap.end()) {
    return position->second;
  } else {
    edm::LogError("LogicError") << "@SUB=RunRangeDependentPedeLabeler::lasBeamIdFromLabel"
				<< "Alignable label " << aliLabel << " not in map.";
    return 0;
  }
}

unsigned int RunRangeDependentPedeLabeler::runRangeIndexFromLabel(unsigned int label) const
{
  Alignable* ali = alignableFromLabel(label);
  unsigned int firstLabel = alignableLabel(ali);
  return (label-firstLabel)/theMaxNumParam;
}

const RunRangeDependentPedeLabeler::RunRange&
RunRangeDependentPedeLabeler::runRangeFromLabel(unsigned int label) const
{
  Alignable* ali = alignableFromLabel(label);

  AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(ali);
  if (positionAli==theAlignableToRunRangeRangeMap.end())
    return theOpenRunRange;
  
  unsigned int firstLabel = alignableLabel(ali);
  unsigned int runRangeIndex = (label-firstLabel)/theParamInstanceOffset;
  unsigned int paramNum = this->paramNumFromLabel(label);

  RunRangeParamMap::const_iterator positionParam = (*positionAli).second.find(paramNum);
  if (positionParam==(*positionAli).second.end()) {
    return theOpenRunRange;
  }
  
  return positionParam->second[runRangeIndex];
}

//__________________________________________________________________________________________________
std::vector<std::string> RunRangeDependentPedeLabeler::decompose(const std::string &s,
								 std::string::value_type delimiter) const
{
  std::vector<std::string> result;

  std::string::size_type previousPos = 0;
  while (true) {
    const std::string::size_type delimiterPos = s.find(delimiter, previousPos);
    if (delimiterPos == std::string::npos) {
      result.push_back(s.substr(previousPos)); // until end
      break;
    }
    result.push_back(s.substr(previousPos, delimiterPos - previousPos));
    previousPos = delimiterPos + 1; // +1: skip delimiter
  }

  return result;
}

//__________________________________________________________________________________________________
std::vector<unsigned int> RunRangeDependentPedeLabeler::convertParamSel(const std::string &selString) const
{
  std::vector<unsigned int> result;
  for (std::string::size_type pos = 0; pos < selString.size(); ++pos) {
    if (selString[pos]=='1') result.push_back(pos);
  }
  return result;
}

unsigned int RunRangeDependentPedeLabeler::buildRunRangeDependencyMap(AlignableTracker *aliTracker,
								      AlignableMuon* aliMuon,
								      AlignableExtras *aliExtras,
								      const edm::ParameterSet &config)
{
  static bool oldRunRangeSelectionWarning = false;

  theAlignableToRunRangeRangeMap.clear();

  AlignmentParameterSelector selector(aliTracker, aliMuon, aliExtras);
 
  std::vector<char> paramSelDummy(6, '1');
  
  const std::vector<edm::ParameterSet> RunRangeSelectionVPSet =
    config.getUntrackedParameter<std::vector<edm::ParameterSet> >("RunRangeSelection");
  
  for (std::vector<edm::ParameterSet>::const_iterator iter = RunRangeSelectionVPSet.begin();
       iter != RunRangeSelectionVPSet.end();
       ++iter) {
    
    const std::vector<std::string> tempRunRanges = (*iter).getParameter<std::vector<std::string> >("RunRanges");
    if (tempRunRanges.size()==0) {
      throw cms::Exception("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
					<< "RunRanges empty\n";
    }

    RunRangeVector RunRanges;
    cond::Time_t first;
    long int temp;
    for (std::vector<std::string>::const_iterator iRunRange = tempRunRanges.begin();
	 iRunRange != tempRunRanges.end();
	 ++iRunRange) {
      if ((*iRunRange).find(':')==std::string::npos) {
	
	first = cond::timeTypeSpecs[cond::runnumber].beginValue;
	temp = strtol((*iRunRange).c_str(), 0, 0);
	if (temp!=-1) first = temp;
	
      } else {
	
	if (!oldRunRangeSelectionWarning) {
	  edm::LogWarning("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap"
				       << "Config file contains old format for 'RunRangeSelection'. Only the start run\n"
				       << "number is used internally. The number of the last run is ignored and can be\n"
				       << "safely removed from the config file.\n";
	  oldRunRangeSelectionWarning = true;
	}

	std::vector<std::string> tokens = edm::tokenize(*iRunRange, ":");
	first = cond::timeTypeSpecs[cond::runnumber].beginValue;
	temp = strtol(tokens[0].c_str(), 0, 0);
	if (temp!=-1) first = temp;
	
      }
      
      RunRanges.push_back(std::pair<cond::Time_t,cond::Time_t>(first, cond::timeTypeSpecs[cond::runnumber].endValue));
    }
    
    for (unsigned int i = 0;i<RunRanges.size()-1;++i) {
      RunRanges[i].second = RunRanges[i+1].first - 1;
      if (RunRanges[i].first > RunRanges[i].second) {
	throw cms::Exception("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
					  << "Inconsistency in 'RunRangeSelection' parameter set.";
      }
    }
    
    const std::vector<std::string> selStrings = (*iter).getParameter<std::vector<std::string> >("selector");
    for (std::vector<std::string>::const_iterator iSel = selStrings.begin();
	 iSel != selStrings.end();
	 ++iSel) {
      std::vector<std::string> decompSel(this->decompose(*iSel, ','));
      
      if (decompSel.size()!=2) {
	throw cms::Exception("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
					  << *iSel <<" should have at least 2 ','-separated parts\n";
      }

      std::vector<unsigned int> selParam = this->convertParamSel(decompSel[1]);
      selector.clear();
      selector.addSelection(decompSel[0], paramSelDummy);

      const std::vector<Alignable*> &alis = selector.selectedAlignables();
       
      for (std::vector<Alignable*>::const_iterator iAli = alis.begin();
	   iAli != alis.end();
	   ++iAli) {
        
        if((*iAli)->alignmentParameters() == NULL) {
          throw cms::Exception("BadConfig")
            << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
            << "Run dependence configured for alignable of type "
            << AlignableObjectId::idToString((*iAli)->alignableObjectId())
            << " at (" << (*iAli)->globalPosition().x() << ","<< (*iAli)->globalPosition().y() << "," << (*iAli)->globalPosition().z()<< "), "
            << "but that has no parameters. Please check that all run "
            << "dependent parameters are also selected for alignment.\n"; 
        }

	for (std::vector<unsigned int>::const_iterator iParam = selParam.begin();
	     iParam != selParam.end();
	     ++iParam) {
	  AlignableToRunRangeRangeMap::const_iterator positionAli = theAlignableToRunRangeRangeMap.find(*iAli);
	  if (positionAli!=theAlignableToRunRangeRangeMap.end()) {
	    
	    AlignmentParameters *AliParams = (*positionAli).first->alignmentParameters();
	    if (static_cast<int>(selParam[selParam.size()-1]) >= AliParams->size()) {
	      throw cms::Exception("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
						<< "mismatch in number of parameters\n";
	    }
	    
	    RunRangeParamMap::const_iterator positionParam = (*positionAli).second.find(*iParam);
	    if (positionParam!=(*positionAli).second.end()) {
	      throw cms::Exception("BadConfig") << "@SUB=RunRangeDependentPedeLabeler::buildRunRangeDependencyMap\n"
						<< "RunRange range for parameter specified twice\n";
	    }
	  }
	  
	  theAlignableToRunRangeRangeMap[*iAli][*iParam] = RunRanges;
	}
      }
    }
  }
  
  return theAlignableToRunRangeRangeMap.size();
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::buildMap(const std::vector<Alignable*> &alis)
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

  if (id > theParamInstanceOffset) { // 'overflow' per instance
    throw cms::Exception("Alignment") << "@SUB=RunRangeDependentPedeLabeler::buildMap: " 
                                      << "Too many labels per instance (" << id-1 << ") leading to double use, "
                                      << "increase PedeLabelerBase::theParamInstanceOffset!\n";
  }
  // return combined size
  return theAlignableToIdMap.size() + theLasBeamToLabelMap.size();
}

//_________________________________________________________________________
unsigned int RunRangeDependentPedeLabeler::buildReverseMap()
{

  // alignables
  theIdToAlignableMap.clear();  // just in case of re-use...

  for (AlignableToIdMap::iterator it = theAlignableToIdMap.begin();
       it != theAlignableToIdMap.end(); ++it) {
    const unsigned int key = (*it).second;
    Alignable *ali = (*it).first;
    const unsigned int nInstances = this->numberOfParameterInstances(ali, -1);
    theMaxNumberOfParameterInstances = std::max(nInstances, theMaxNumberOfParameterInstances);
    for (unsigned int iInstance=0;iInstance<nInstances;++iInstance) {
      theIdToAlignableMap[key+iInstance*theParamInstanceOffset] = ali;
    }
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
DEFINE_EDM_PLUGIN(PedeLabelerPluginFactory, RunRangeDependentPedeLabeler, "RunRangeDependentPedeLabeler");
