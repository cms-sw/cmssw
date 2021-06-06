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
#include <vector>

class CSCLUTReader;
class CSCALCTDigi;
class CSCCLCTDigi;

class CSCALCTCrossCLCT {
public:
  CSCALCTCrossCLCT(unsigned endcap, unsigned station, unsigned ring, bool isganged, const edm::ParameterSet& luts);

  // check if an ALCT can cross a CLCT. Not always the case for ME1/1
  bool doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c, bool ignoreAlctCrossClct) const;

private:
  bool doesWiregroupCrossHalfStrip(int wg, int keystrip) const;

  unsigned endcap_;
  unsigned station_;
  unsigned ring_;
  bool isganged_;

  // strings to paths of LUTs
  std::vector<std::string> wgCrossHsME1aFiles_;
  std::vector<std::string> wgCrossHsME1aGangedFiles_;
  std::vector<std::string> wgCrossHsME1bFiles_;

  // unique pointers to the luts
  std::unique_ptr<CSCLUTReader> wg_cross_min_hs_ME1a_;
  std::unique_ptr<CSCLUTReader> wg_cross_max_hs_ME1a_;
  std::unique_ptr<CSCLUTReader> wg_cross_min_hs_ME1a_ganged_;
  std::unique_ptr<CSCLUTReader> wg_cross_max_hs_ME1a_ganged_;
  std::unique_ptr<CSCLUTReader> wg_cross_min_hs_ME1b_;
  std::unique_ptr<CSCLUTReader> wg_cross_max_hs_ME1b_;
};

#endif
