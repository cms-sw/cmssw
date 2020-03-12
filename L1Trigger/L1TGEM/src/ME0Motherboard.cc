#include "L1Trigger/L1TGEM/interface/ME0Motherboard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

ME0Motherboard::ME0Motherboard(unsigned endcap, unsigned chamber, const edm::ParameterSet& conf)
    : theEndcap(endcap), theChamber(chamber) {
  edm::ParameterSet tmbParams = conf.getParameter<edm::ParameterSet>("tmbParam");
  infoV = tmbParams.getParameter<int>("verbosity");
}

ME0Motherboard::ME0Motherboard() : theEndcap(1), theChamber(1) { infoV = 2; }

ME0Motherboard::~ME0Motherboard() {}

void ME0Motherboard::clear() {
  for (int bx = 0; bx < MAX_TRIGGER_BINS; bx++) {
    for (int i = 0; i < MAX_TRIGGERS; i++) {
      Triggers[bx][i].clear();
    }
  }
}

void ME0Motherboard::run(const ME0PadDigiCollection*) { clear(); }

// Returns vector of read-out correlated Triggers, if any.  Starts with
// the vector of all found Triggers and selects the ones in the read-out
// time window.
std::vector<ME0TriggerDigi> ME0Motherboard::readoutTriggers() {
  std::vector<ME0TriggerDigi> tmpV;

  std::vector<ME0TriggerDigi> all_trigs = getTriggers();
  for (const auto& ptrig : all_trigs) {
    // in the future, add a selection on the BX
    tmpV.push_back(ptrig);
  }
  return tmpV;
}

// Returns vector of all found correlated Triggers, if any.
std::vector<ME0TriggerDigi> ME0Motherboard::getTriggers() {
  std::vector<ME0TriggerDigi> tmpV;

  for (int bx = 0; bx < MAX_TRIGGER_BINS; bx++) {
    for (int i = 0; i < MAX_TRIGGERS; i++) {
      tmpV.push_back(Triggers[bx][i]);
    }
  }
  return tmpV;
}

// compare Triggers by quality
bool ME0Motherboard::sortByQuality(const ME0TriggerDigi& trig1, const ME0TriggerDigi& trig2) {
  return trig1.getQuality() > trig2.getQuality();
}

// compare Triggers by GEM bending angle
bool ME0Motherboard::sortByME0Dphi(const ME0TriggerDigi& trig1, const ME0TriggerDigi& trig2) {
  // todo: In the future I plan a member to be added to ME0TriggerDigi, getME0Dphi().
  // That function will derive the bending angle from the pattern.
  // The ME0TriggerDigi pattterns are at this point not defined yet.
  return true;
}
