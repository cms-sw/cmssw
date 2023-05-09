#ifndef L1Trigger_CSCTriggerPrimitives_ComparatorCodeLUT
#define L1Trigger_CSCTriggerPrimitives_ComparatorCodeLUT

/** \class ComparatorCodeLUT
 *
 * Helper class to calculate for the comparator code
 * algorithm for Phase-2.
 *
 * \author Sven Dildick (Rice University)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCPatternBank.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableCCLUT.h"

#include <vector>
#include <string>

class CSCCLCTDigi;

class ComparatorCodeLUT {
public:
  typedef std::array<std::array<int, 3>, CSCConstants::NUM_LAYERS> pattern;

  // constructor
  ComparatorCodeLUT(const edm::ParameterSet& conf);

  // runs the CCLUT procedure
  void run(CSCCLCTDigi& digi, unsigned numCFEBs, const CSCL1TPLookupTableCCLUT* lookupTableCCLUT) const;

private:
  //calculates the id based on location of hits
  int calculateComparatorCode(const pattern& halfStripPattern) const;

  unsigned convertSlopeToRun2Pattern(const unsigned slope) const;

  // sets the 1/4 and 1/8 strip bits given a floating point position offset
  void assignPositionCC(const unsigned offset, std::tuple<int16_t, bool, bool>& returnValue) const;

  // actual LUT used
  CSCPatternBank::LCTPatterns clct_pattern_ = {};

  // verbosity level
  unsigned infoV_;
};

#endif
