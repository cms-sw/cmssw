#ifndef L1Trigger_ME0TriggerPrimitives_ME0Motherboard_h
#define L1Trigger_ME0TriggerPrimitives_ME0Motherboard_h

#include "DataFormats/GEMDigi/interface/ME0LCTDigi.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ME0Motherboard
{
 public:
  /** Normal constructor. */
  ME0Motherboard(unsigned endcap, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  ME0Motherboard();

  /** Default destructor. */
  ~ME0Motherboard();

  /** Run function for normal usage. */
  void run(const ME0PadDigiCollection*);

  /** Returns vector of LCTs in the read-out time window, if any. */
  std::vector<ME0LCTDigi> readoutLCTs();

  /** Returns vector of all found correlated LCTs, if any. */
  std::vector<ME0LCTDigi> getLCTs();

  /** Clears LCTs. */
  void clear();

 private:

  /** Verbosity level: 0: no print (default).
   *                   1: print LCTs found. */
  int infoV;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theChamber;

  /** Maximum number of time bins. */
  enum {MAX_LCT_BINS = 1, MAX_LCTS = 8};

  /** Container for LCTs. */
  ME0LCTDigi LCTs[MAX_LCT_BINS][MAX_LCTS];

  // utilities for sorting
  static bool sortByQuality(const ME0LCTDigi&, const ME0LCTDigi&); 
  static bool sortByME0Dphi(const ME0LCTDigi&, const ME0LCTDigi&); 
};
#endif
