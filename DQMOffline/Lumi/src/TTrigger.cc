#include "DQMOffline/Lumi/interface/TTrigger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <limits> 

using namespace ZCountingTrigger;

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
TTrigger::TTrigger(const std::vector<std::string> &muonTriggerNames, const std::vector<std::string> &muonTriggerObjectNames) { 

  for(unsigned int i = 0; i < muonTriggerNames.size(); ++i){
    fRecords.push_back(ZCountingTrigger::TriggerRecord(muonTriggerNames.at(i),0));
    fRecords.back().objectMap.push_back(std::pair<std::string, int>(muonTriggerObjectNames.at(i),0));    
  }
  fRecords.push_back(ZCountingTrigger::TriggerRecord("HLT_Ele35_WPTight_Gsf_v*",1));
  fRecords.back().objectMap.push_back(std::pair<std::string, int>("hltEle35noerWPTightGsfTrackIsoFilter",0));
  fRecords.push_back(ZCountingTrigger::TriggerRecord("HLT_Ele27_WPTight_Gsf_v*",2));
  fRecords.back().objectMap.push_back(std::pair<std::string, int>("hltEle27WPTightGsfTrackIsoFilter",0));
}

//--------------------------------------------------------------------------------------------------
int TTrigger::getTriggerBit(const std::string &iName) const { 
  int lId = -1;
  for(unsigned int i0 = 0; i0 < fRecords.size(); i0++) { 
    if(iName == fRecords[i0].hltPattern) lId = i0;    
  }
  if(lId == -1) edm::LogWarning("ZCounting") << "=== Missing Trigger ==" << iName << std::endl;
  return lId;
}

//--------------------------------------------------------------------------------------------------
int TTrigger::getTriggerObjectBit(const std::string &iName, const std::string &iObjName) const {
  int lId = getTriggerBit(iName);
  if(lId == -1) return -1;

  for(unsigned int i0 = 0; i0 < fRecords[lId].objectMap.size(); i0++) {
    if(iObjName != fRecords[lId].objectMap[i0].first) continue;
    return fRecords[lId].objectMap[i0].second;
  }

  return -1;
}

//--------------------------------------------------------------------------------------------------
bool TTrigger::pass(const std::string &iName, const TriggerBits &iTrig) const {
  int lId = getTriggerBit(iName);
  if(lId == -1) return false;

  return iTrig[lId];
}

//--------------------------------------------------------------------------------------------------
bool TTrigger::passObj(const std::string &iName, const std::string &iObjName, const TriggerObjects &iTrigObj) const {
  int lId = getTriggerObjectBit(iName,iObjName);
  if(lId == -1) return false;

  return iTrigObj[lId];
}
