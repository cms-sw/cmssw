#include "DQMOffline/LumiZCounting/interface/TTrigger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <limits> 

using namespace baconhep;

//--------------------------------------------------------------------------------------------------
//
// Reads in an input file specifying the triggers we're interested in.
// The input file has the format:
//  <trigger name>   <trigger object name>   <trigger object leg #>
//
// A trigger with multiple objects will have multiple entries with the same <trigger name> but
// one entry for each <trigger object name>.
//
// The trigger object leg numbering is to account for the possibility that a particular object of
// the trigger can evolve and obtain a different trigger object name, but we still want this to 
// be associated with the same leg (e.g. the trailing electron in a dielectron trigger)
//
TTrigger::TTrigger(const std::string iFileName) { 
  std::ifstream lFile(iFileName.c_str());
  assert(lFile.is_open());

  int         lTrigIndex    = -1;  // bacon trigger bits index
  int         lTrigObjIndex = -1;  // bacon trigger object bits index
  std::string lLastTrigName = "";  // keep track of when new trigger name is encountered
  int         lLastLeg      = -1;  // keep track of when new trigger object leg is encountered

  while(lFile.good()) { 
    std::string pTrigName; 
    std::string pFilterName;
    int         pLeg;

    lFile >> pTrigName;

    // allow C++ style comments in HLT file
    if(pTrigName.find("//") != std::string::npos) {
      lFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    // ignore empty lines
    if(pTrigName == "") continue;

    lFile >> pFilterName >> pLeg;

    if(lLastTrigName != pTrigName) {
      lTrigIndex++;
      fRecords.push_back(baconhep::TriggerRecord(pTrigName,lTrigIndex));
      lTrigObjIndex++;

    } else if(lLastLeg != pLeg) {
      lTrigObjIndex++;
    }

    fRecords.back().objectMap.push_back(std::pair<std::string, int>(pFilterName,lTrigObjIndex));
    lLastTrigName = pTrigName;
    lLastLeg      = pLeg;

    // ignore all trailing stuff
    lFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
}

//--------------------------------------------------------------------------------------------------
int TTrigger::getTriggerBit(const std::string iName) const { 
  int lId = -1;
  for(unsigned int i0 = 0; i0 < fRecords.size(); i0++) { 
    if(iName == fRecords[i0].hltPattern) lId = i0;    
  }
  if(lId == -1) edm::LogWarning("ZCounting") << "=== Missing Trigger ==" << iName << std::endl;
  return lId;
}

//--------------------------------------------------------------------------------------------------
int TTrigger::getTriggerObjectBit(const std::string iName, const int iLeg) const { 
  int lId = getTriggerBit(iName);
  if(lId == -1) return -1;

  //
  // Assumes trigger object legs are numbered 1,2,3, etc.
  // The logic below looks for index changes in the objectMap.
  // For example, if looking for the trigger object bit for leg 3, looping through the objectMap
  // list one needs to find the 3rd unique object bit associated with the trigger record.
  //
  if(iLeg==1) {
    return fRecords[lId].objectMap[0].second;
  }
  int tmp = 1;
  unsigned int lastObjIndex = fRecords[lId].objectMap[0].second;
  for(unsigned int i0 = 1; i0 < fRecords[lId].objectMap.size(); i0++) {
    unsigned int newId = fRecords[lId].objectMap[i0].second;
    if(lastObjIndex != newId) {
      lastObjIndex = newId;
      tmp++;
      if(tmp==iLeg) {
        return fRecords[lId].objectMap[i0].second;
      }
    }
  }

  return -1;
}

//--------------------------------------------------------------------------------------------------
int TTrigger::getTriggerObjectBit(const std::string iName, const std::string iObjName) const {
  int lId = getTriggerBit(iName);
  if(lId == -1) return -1;

  for(unsigned int i0 = 0; i0 < fRecords[lId].objectMap.size(); i0++) {
    if(iObjName != fRecords[lId].objectMap[i0].first) continue;
    return fRecords[lId].objectMap[i0].second;
  }

  return -1;
}

//--------------------------------------------------------------------------------------------------
bool TTrigger::pass(const std::string iName, const TriggerBits &iTrig) const {
  int lId = getTriggerBit(iName);
  if(lId == -1) return false;

  return iTrig[lId];
}

//--------------------------------------------------------------------------------------------------
bool TTrigger::passObj(const std::string iName, const int iLeg, const TriggerObjects &iTrigObj) const {
  int lId = getTriggerObjectBit(iName,iLeg);
  if(lId == -1) return false;

  return iTrigObj[lId];
}

//--------------------------------------------------------------------------------------------------
bool TTrigger::passObj(const std::string iName, const std::string iObjName, const TriggerObjects &iTrigObj) const {
  int lId = getTriggerObjectBit(iName,iObjName);
  if(lId == -1) return false;

  return iTrigObj[lId];
}
