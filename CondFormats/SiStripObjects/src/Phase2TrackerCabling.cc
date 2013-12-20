#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerModule.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>


const Phase2TrackerCabling& Phase2TrackerCabling::fedCabling() const {
  // check the mode (current sorting)
  if (mode_!=1) {
    // sort according to the fed id/ch
    std::sort(connections_.begin(),connections_.end(),Phase2TrackerModule::chOrdering);
    // set the new mode
    mode_=1;
  }
  // return itself as an object reference -> allows to chain with find.
  return *this;
}

const Phase2TrackerCabling& Phase2TrackerCabling::detCabling() const {
  // check the mode (current sorting)
  if (mode_!=2) {
    // sort according to the detid
    std::sort(connections_.begin(),connections_.end(),Phase2TrackerModule::detidOrdering);
    // set the new mode
    mode_=2;
  }
  // return itself as an object reference -> allows to chain with find.
  return *this;
}

const Phase2TrackerCabling& Phase2TrackerCabling::gbtCabling() const {
  // check the mode (current sorting)
  if (mode_!=3) {
    // sort according to the gbtid (construction map)
    std::sort(connections_.begin(),connections_.end(),Phase2TrackerModule::gbtidOrdering);
    // set the new mode
    mode_=3;
  }
  // return itself as an object reference -> allows to chain with find.
  return *this;
}

void Phase2TrackerCabling::checkMode(const char* funcname, int mode) const {
  if(mode_!=mode) {
    std::string message = std::string(funcname);
    switch(mode_) {
      case 0:
        message += " called on a unsorted cabling. ";
        break;
      case 1:
        message += " called on a cabling prepared for fed ids. ";
        break;
      case 2:
        message += " called on a cabling prepared for det ids. ";
        break;
      case 3:
        message += " called on a cabling prepared for gbt ids. ";
        break;
      default:
        message += " called on a badly defined cabling. ";
    }
    switch(mode) {
      case 1:
        message += "Calling Phase2TrackerCabling::fedCabling() first.";
        fedCabling();
        break;
      case 2:
        message += "Calling Phase2TrackerCabling::detCabling() first.";
        detCabling();
        break;
      case 3:
        message += "Calling Phase2TrackerCabling::gbtCabling() first.";
        gbtCabling();
        break;
      default:
        message += "Unknown mode. No way to switch to it.";
    }
    edm::LogWarning("UnsortedCabling") << message;
  } 
  else return;
} 

const Phase2TrackerModule& Phase2TrackerCabling::findFedCh(std::pair<unsigned int, unsigned int> fedch) const {
  // check the proper mode
  checkMode(__PRETTY_FUNCTION__,1);
  // look for ch
  std::vector<Phase2TrackerModule>::const_iterator itid = std::lower_bound (connections_.begin(), connections_.end(), fedch, Phase2TrackerModule::chComp);
  if (itid != connections_.end() && itid->getCh()==fedch)
    return *itid;
  else
    throw cms::Exception("IndexNotFound") << "No connection corresponding to FED id/ch = " << fedch.first << "/" << fedch.second;
}

const Phase2TrackerModule& Phase2TrackerCabling::findDetid(uint32_t detid) const {
  // check the proper mode
  checkMode(__PRETTY_FUNCTION__,2);
  // look for id 
  std::vector<Phase2TrackerModule>::const_iterator itch = std::lower_bound (connections_.begin(), connections_.end(), detid, Phase2TrackerModule::detidComp);
  if (itch != connections_.end() && itch->getDetid()==detid)
    return *itch;
  else
    throw cms::Exception("IndexNotFound") << "No connection corresponding to detid = 0x" << std::hex << detid << std::dec;
}

const Phase2TrackerModule& Phase2TrackerCabling::findGbtid(uint32_t gbtid) const {
  // check the proper mode
  checkMode(__PRETTY_FUNCTION__,3);
  // look for id 
  std::vector<Phase2TrackerModule>::const_iterator itch = std::lower_bound (connections_.begin(), connections_.end(), gbtid, Phase2TrackerModule::gbtidComp);
  if (itch != connections_.end() && itch->getGbtid()==gbtid)
    return *itch;
  else
    throw cms::Exception("IndexNotFound") << "No connection corresponding to gbtid = 0x" << std::hex << gbtid << std::dec;
}

Phase2TrackerCabling Phase2TrackerCabling::filterByCoolingLine(uint32_t coolingLine) const {
  // sort according to cooling
  std::sort(connections_.begin(),connections_.end(),Phase2TrackerModule::coolingOrdering);
  // search for the proper range
  std::pair< std::vector<Phase2TrackerModule>::const_iterator, std::vector<Phase2TrackerModule>::const_iterator > range = std::equal_range(connections_.begin(),connections_.end(),Phase2TrackerModule(Phase2TrackerModule::SS,0,0,0,0,0,coolingLine),Phase2TrackerModule::coolingOrdering);
  // create a new cabling object
  Phase2TrackerCabling result(std::vector<Phase2TrackerModule>(range.first,range.second));
  // restore the initial ordering
  switch(mode_) {
    case 1:
      fedCabling();
      break;
    case 2:
      detCabling();
      break;
    case 3:
      gbtCabling();
      break;
  }
  // return the new cabling object
  return result;
}

Phase2TrackerCabling Phase2TrackerCabling::filterByPowerGroup(uint32_t powerGroup) const {
  // sort according to power groups
  std::sort(connections_.begin(),connections_.end(),Phase2TrackerModule::powerOrdering);
  // search for the proper range
  std::pair< std::vector<Phase2TrackerModule>::const_iterator, std::vector<Phase2TrackerModule>::const_iterator > range = std::equal_range(connections_.begin(),connections_.end(),Phase2TrackerModule(Phase2TrackerModule::SS,0,0,0,0,powerGroup,0),Phase2TrackerModule::powerOrdering);
  // create a new cabling object
  Phase2TrackerCabling result(std::vector<Phase2TrackerModule>(range.first,range.second));
  // restore the initial ordering
  switch(mode_) {
    case 1:
      fedCabling();
      break;
    case 2:
      detCabling();
      break;
    case 3:
      gbtCabling();
      break;
  }
  // return the new cabling object
  return result;
}

bool fedeq (Phase2TrackerModule& a, Phase2TrackerModule& b) {
  return (a.getCh().first==b.getCh().first);
}

bool cooleq (Phase2TrackerModule& a, Phase2TrackerModule& b) {
  return (a.getCoolingLoop()==b.getCoolingLoop());
}

bool poweq (Phase2TrackerModule& a, Phase2TrackerModule& b) {
  return (a.getPowerGroup()==b.getPowerGroup());
}

std::string Phase2TrackerCabling::summaryDescription() const {
  std::string mystring("Summary of the cabling\n======================\n");
  std::stringstream ss;
  // number of modules, feds, cooling loop and power groups
  ss << "Number of modules: " << connections_.size() << std::endl;
  std::vector<Phase2TrackerModule> orig(connections_);
  ss << "Number of FEDs: ";
  std::sort(orig.begin(),orig.end(),Phase2TrackerModule::chOrdering);
  std::vector<Phase2TrackerModule> tmp(orig);
  ss << std::distance(tmp.begin(),std::unique(tmp.begin(),tmp.end(),fedeq)) << std::endl;
  ss << "Number of cooling loops: ";
  std::sort(orig.begin(),orig.end(),Phase2TrackerModule::coolingOrdering);
  tmp = orig;
  ss << std::distance(tmp.begin(),std::unique(tmp.begin(),tmp.end(),cooleq)) << std::endl;
  ss << "Number of power groups: ";
  std::sort(orig.begin(),orig.end(),Phase2TrackerModule::powerOrdering);
  tmp = orig;
  ss << std::distance(tmp.begin(),std::unique(tmp.begin(),tmp.end(),poweq)) << std::endl;
  mystring += ss.str();
  return mystring;
}

std::string Phase2TrackerCabling::description(bool compact) const {
  std::string mystring("Cabling:\n========\n");
  for(std::vector<Phase2TrackerModule>::const_iterator it = connections_.begin();it<connections_.end();++it) {
    mystring += it->description(compact);
  }
  return mystring;
}

