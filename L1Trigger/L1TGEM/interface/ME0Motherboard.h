#ifndef L1Trigger_L1TGEM_ME0Motherboard_h
#define L1Trigger_L1TGEM_ME0Motherboard_h

/** \class ME0TriggerBuilder
 *
 * Does pattern recognition of ME0 pads to build ME0 triggers
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ME0Geometry;

class ME0Motherboard {
public:
  /** Normal constructor. */
  ME0Motherboard(unsigned endcap, unsigned chamber, const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  ME0Motherboard();

  /** Default destructor. */
  ~ME0Motherboard();

  /** set geometry for the matching needs */
  void setME0Geometry(const ME0Geometry* g) { me0_g = g; }

  /** Run function for normal usage. */
  void run(const ME0PadDigiCollection*);

  /** Returns vector of Triggers in the read-out time window, if any. */
  std::vector<ME0TriggerDigi> readoutTriggers();

  /** Returns vector of all found correlated Triggers, if any. */
  std::vector<ME0TriggerDigi> getTriggers();

  /** Clears Triggers. */
  void clear();

private:
  /** Verbosity level: 0: no print (default).
   *                   1: print Triggers found. */
  int infoV;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theChamber;

  const ME0Geometry* me0_g;

  /** Maximum number of time bins. */
  enum { MAX_TRIGGER_BINS = 1, MAX_TRIGGERS = 8 };

  /** Container for Triggers. */
  ME0TriggerDigi Triggers[MAX_TRIGGER_BINS][MAX_TRIGGERS];

  // utilities for sorting
  static bool sortByQuality(const ME0TriggerDigi&, const ME0TriggerDigi&);
  static bool sortByME0Dphi(const ME0TriggerDigi&, const ME0TriggerDigi&);
};
#endif
