#ifndef L1Trigger_CSCTriggerPrimitives_CSCALCTCrossCLCT
#define L1Trigger_CSCTriggerPrimitives_CSCALCTCrossCLCT

/** \class CSCALCTCrossCLCT
 *
 * Helper class to check if an ALCT crosses with a CLCT in ME1/1.
 * This check was originally introduced by Vadim Khotilovich in 2010
 * to improve the quality of the ME1/1 LCTs that are sent to the
 * endcap muon track finder. However, it is the policy of the EMTF
 * group that they would like to receive all LCT information, even
 * if an ME1/1 LCT has no physical crossing half-strips and
 * wiregroups. The EMTF disassembles LCTs into ALCT and CLCT anyway
 * and thus for normal trigger operations this class should not be
 * used. However, in the event that multiple high-quality LCTs are
 * present in ME1/1, this class could be used to trim the unphysical
 * ones. As of writing (April 2021) no plans are in place to implement
this feature into the CSC trigger firmware
 *
 * \author Sven Dildick (Rice University)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <array>

class CSCALCTDigi;
class CSCCLCTDigi;

class CSCALCTCrossCLCT {
public:
  CSCALCTCrossCLCT(
      unsigned endcap, unsigned station, unsigned ring, bool ignoreAlctCrossClct, const edm::ParameterSet& conf);

  /*
    Check if an ALCT can cross a CLCT. Most of the time it can. Only in ME1/1 there are
    special cases when they do not. This function is typically not used though, as the
    EMTF prefers to receive all stubs. However, there is an option to discard unphysical matches.
  */
  bool doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c) const;

private:
  // check if a wiregroup cross a halfstrip
  bool doesWiregroupCrossHalfStrip(int wg, int keystrip) const;

  unsigned endcap_;
  unsigned station_;
  unsigned ring_;
  bool gangedME1a_;
  bool ignoreAlctCrossClct_;
};
#endif
