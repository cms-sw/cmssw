#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerModule.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

// These functions are used to sort and search the cabling objects

// by FED id/ch
bool Phase2TrackerCabling::chOrdering(Phase2TrackerCabling::key a, Phase2TrackerCabling::key b) {
  if (a->getCh().first == b->getCh().first)
    return a->getCh().second < b->getCh().second;
  else
    return a->getCh().first < b->getCh().first;
}
bool Phase2TrackerCabling::chComp(Phase2TrackerCabling::key a, std::pair<unsigned int, unsigned int> b) {
  if (a->getCh().first == b.first)
    return a->getCh().second < b.second;
  else
    return a->getCh().first < b.first;
}
bool Phase2TrackerCabling::fedeq(key a, key b) { return (a->getCh().first == b->getCh().first); }

// by detid
bool Phase2TrackerCabling::detidOrdering(Phase2TrackerCabling::key a, Phase2TrackerCabling::key b) {
  return a->getDetid() < b->getDetid();
}
bool Phase2TrackerCabling::detidComp(Phase2TrackerCabling::key a, uint32_t b) { return a->getDetid() < b; }

// by gbtid
bool Phase2TrackerCabling::gbtidOrdering(Phase2TrackerCabling::key a, Phase2TrackerCabling::key b) {
  return a->getGbtid() < b->getGbtid();
}
bool Phase2TrackerCabling::gbtidComp(Phase2TrackerCabling::key a, uint32_t b) { return a->getGbtid() < b; }

// by cooling loop
bool Phase2TrackerCabling::coolingOrdering(const Phase2TrackerModule& a, const Phase2TrackerModule& b) {
  return a.getCoolingLoop() < b.getCoolingLoop();
}
bool Phase2TrackerCabling::coolingComp(const Phase2TrackerModule& a, uint32_t b) { return a.getCoolingLoop() < b; }
bool Phase2TrackerCabling::cooleq(const Phase2TrackerModule& a, const Phase2TrackerModule& b) {
  return (a.getCoolingLoop() == b.getCoolingLoop());
}

// by power group
bool Phase2TrackerCabling::powerOrdering(const Phase2TrackerModule& a, const Phase2TrackerModule& b) {
  return a.getPowerGroup() < b.getPowerGroup();
}
bool Phase2TrackerCabling::powerComp(const Phase2TrackerModule& a, uint32_t b) { return a.getPowerGroup() < b; }
bool Phase2TrackerCabling::poweq(const Phase2TrackerModule& a, const Phase2TrackerModule& b) {
  return (a.getPowerGroup() == b.getPowerGroup());
}

// Phase2TrackerCabling methods

Phase2TrackerCabling::Phase2TrackerCabling(const std::vector<Phase2TrackerModule>& cons) : connections_(cons) {
  // initialize the cabling (fill the transient objects and sort.
  initializeCabling();
}

Phase2TrackerCabling::Phase2TrackerCabling(const Phase2TrackerCabling& src) {
  connections_ = src.connections_;
  fedCabling_ = src.fedCabling_;
  detCabling_ = src.detCabling_;
  gbtCabling_ = src.gbtCabling_;
}

void Phase2TrackerCabling::initializeCabling() {
  // fill the cabling objects
  fedCabling_.reserve(connections_.size());
  detCabling_.reserve(connections_.size());
  gbtCabling_.reserve(connections_.size());
  for (key module = connections_.begin(); module < connections_.end(); ++module) {
    fedCabling_.push_back(module);
    detCabling_.push_back(module);
    gbtCabling_.push_back(module);
  }
  // sort the cabling objects
  std::sort(fedCabling_.begin(), fedCabling_.end(), chOrdering);
  std::sort(detCabling_.begin(), detCabling_.end(), detidOrdering);
  std::sort(gbtCabling_.begin(), gbtCabling_.end(), gbtidOrdering);
}

const Phase2TrackerModule& Phase2TrackerCabling::findFedCh(std::pair<unsigned int, unsigned int> fedch) const {
  // look for ch
  cabling::const_iterator itid = std::lower_bound(fedCabling_.begin(), fedCabling_.end(), fedch, chComp);
  if (itid != fedCabling_.end() && (*itid)->getCh() == fedch)
    return **itid;
  else
    throw cms::Exception("IndexNotFound")
        << "No connection corresponding to FED id/ch = " << fedch.first << "/" << fedch.second;
}

const Phase2TrackerModule& Phase2TrackerCabling::findDetid(uint32_t detid) const {
  // look for id
  cabling::const_iterator itch = std::lower_bound(detCabling_.begin(), detCabling_.end(), detid, detidComp);
  if (itch != detCabling_.end() && (*itch)->getDetid() == detid)
    return **itch;
  else
    throw cms::Exception("IndexNotFound")
        << "No connection corresponding to detid = 0x" << std::hex << detid << std::dec;
}

const Phase2TrackerModule& Phase2TrackerCabling::findGbtid(uint32_t gbtid) const {
  // look for id
  cabling::const_iterator itch = std::lower_bound(gbtCabling_.begin(), gbtCabling_.end(), gbtid, gbtidComp);
  if (itch != gbtCabling_.end() && (*itch)->getGbtid() == gbtid)
    return **itch;
  else
    throw cms::Exception("IndexNotFound")
        << "No connection corresponding to gbtid = 0x" << std::hex << gbtid << std::dec;
}

Phase2TrackerCabling Phase2TrackerCabling::filterByCoolingLine(uint32_t coolingLine) const {
  // NB: this approach involves two copies of the connections. Can we do better?
  // since this is a relatively rare operation, I don't want to pre-sort the connections.

  // make a copy of the store
  store resultStore = connections_;
  // sort according to cooling
  std::sort(resultStore.begin(), resultStore.end(), coolingOrdering);
  // search for the proper range
  std::pair<key, key> range = std::equal_range(resultStore.begin(),
                                               resultStore.end(),
                                               Phase2TrackerModule(Phase2TrackerModule::SS, 0, 0, 0, 0, 0, coolingLine),
                                               coolingOrdering);
  // create a new cabling object
  Phase2TrackerCabling result(store(range.first, range.second));
  // return the new cabling object
  return result;
}

Phase2TrackerCabling Phase2TrackerCabling::filterByPowerGroup(uint32_t powerGroup) const {
  // NB: this approach involves two copies of the connections. Can we do better?
  // since this is a relatively rare operation, I don't want to pre-sort the connections.

  // make a copy of the store
  store resultStore = connections_;
  // sort according to power groups
  std::sort(resultStore.begin(), resultStore.end(), powerOrdering);
  // search for the proper range
  std::pair<key, key> range = std::equal_range(resultStore.begin(),
                                               resultStore.end(),
                                               Phase2TrackerModule(Phase2TrackerModule::SS, 0, 0, 0, 0, powerGroup, 0),
                                               powerOrdering);
  // create a new cabling object
  Phase2TrackerCabling result(store(range.first, range.second));
  // return the new cabling object
  return result;
}

std::string Phase2TrackerCabling::summaryDescription() const {
  std::string mystring("Summary of the cabling\n======================\n");
  std::stringstream ss;
  // number of modules, feds, cooling loop and power groups
  ss << "Number of modules: " << connections_.size() << std::endl;
  store orig(connections_);
  ss << "Number of FEDs: ";
  cabling tmpc(fedCabling_);
  ss << std::distance(tmpc.begin(), std::unique(tmpc.begin(), tmpc.end(), fedeq)) << std::endl;
  ss << "Number of cooling loops: ";
  std::sort(orig.begin(), orig.end(), coolingOrdering);
  store tmp(orig);
  ss << std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end(), cooleq)) << std::endl;
  ss << "Number of power groups: ";
  std::sort(orig.begin(), orig.end(), powerOrdering);
  tmp = orig;
  ss << std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end(), poweq)) << std::endl;
  mystring += ss.str();
  return mystring;
}

std::string Phase2TrackerCabling::description(bool compact) const {
  std::string mystring("Cabling:\n========\n");
  for (std::vector<Phase2TrackerModule>::const_iterator it = connections_.begin(); it < connections_.end(); ++it) {
    mystring += it->description(compact);
  }
  return mystring;
}
