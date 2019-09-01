#ifndef CondFormats_SiStripObjects_Phase2TrackerCabling_H
#define CondFormats_SiStripObjects_Phase2TrackerCabling_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include <CondFormats/SiStripObjects/interface/Phase2TrackerModule.h>
#include <vector>
#include <algorithm>

class Phase2TrackerCabling {
  typedef std::vector<Phase2TrackerModule> store;
  typedef std::vector<Phase2TrackerModule>::const_iterator key;
  typedef std::vector<key> cabling;

public:
  // Constructor taking FED channel connection objects as input.
  Phase2TrackerCabling(const std::vector<Phase2TrackerModule>& cons);

  // Copy ocnstructor
  Phase2TrackerCabling(const Phase2TrackerCabling& src);

  // Default constructor
  Phase2TrackerCabling() {}

  // Default destructor
  virtual ~Phase2TrackerCabling() {}

  // Initialize the internal maps
  void initializeCabling();

  // get the list of modules
  const std::vector<Phase2TrackerModule>& connections() const { return connections_; }

  // find a connection for a given fed channel
  const Phase2TrackerModule& findFedCh(std::pair<unsigned int, unsigned int> fedch) const;

  // find a connection for a given detid
  const Phase2TrackerModule& findDetid(uint32_t detid) const;

  // find a connection for a given gbtid
  const Phase2TrackerModule& findGbtid(uint32_t gbtid) const;

  // return all the modules connected to a given cooling line
  Phase2TrackerCabling filterByCoolingLine(uint32_t coolingLine) const;

  // return all the modules connected to a given HV group
  Phase2TrackerCabling filterByPowerGroup(uint32_t powerGroup) const;

  // print a summary of the content
  std::string summaryDescription() const;

  // print the details of the content
  std::string description(bool compact = false) const;

private:
  // the connections
  store connections_;

  // indices for fast searches
  cabling fedCabling_ COND_TRANSIENT;
  cabling gbtCabling_ COND_TRANSIENT;
  cabling detCabling_ COND_TRANSIENT;

private:
  // sorting functions
  static bool chOrdering(key a, key b);
  static bool chComp(key a, std::pair<unsigned int, unsigned int> b);
  static bool fedeq(key a, key b);
  static bool detidOrdering(key a, key b);
  static bool detidComp(key a, uint32_t b);
  static bool gbtidOrdering(key a, key b);
  static bool gbtidComp(key a, uint32_t b);
  static bool coolingOrdering(const Phase2TrackerModule& a, const Phase2TrackerModule& b);
  static bool coolingComp(const Phase2TrackerModule& a, uint32_t b);
  static bool cooleq(const Phase2TrackerModule& a, const Phase2TrackerModule& b);
  static bool powerOrdering(const Phase2TrackerModule& a, const Phase2TrackerModule& b);
  static bool powerComp(const Phase2TrackerModule& a, uint32_t b);
  static bool poweq(const Phase2TrackerModule& a, const Phase2TrackerModule& b);

  COND_SERIALIZABLE;
};

#endif
