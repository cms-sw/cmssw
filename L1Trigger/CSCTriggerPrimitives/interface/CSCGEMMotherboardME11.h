#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboardME11_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboardME11_h

/** \class CSCGEMMotherboardME11
 *
 * Extended CSCMotherboard for ME11 TMB upgrade
 *
 * \author Sven Dildick March 2014
 *
 * Based on CSCMotherboard code
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboard.h"

class CSCGEMMotherboardME11 : public CSCGEMMotherboard {
public:
  /** Normal constructor. */
  CSCGEMMotherboardME11(unsigned endcap,
                        unsigned station,
                        unsigned sector,
                        unsigned subsector,
                        unsigned chamber,
                        const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCGEMMotherboardME11();

  /** Default destructor. */
  ~CSCGEMMotherboardME11() override;

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc,
           const GEMPadDigiClusterCollection* gemPads) override;

  /* readout the LCTs in ME1a or ME1b */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1a() const;
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1b() const;

private:
  /* access to the LUTs needed for matching */
  const CSCGEMMotherboardLUTME11* getLUT() const override { return tmbLUT_.get(); }
  std::unique_ptr<CSCGEMMotherboardLUTME11> tmbLUT_;
  std::unique_ptr<CSCMotherboardLUTME11> cscTmbLUT_;

  /* readout the LCTs in a sector of ME11 */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTsME11(enum CSCPart me1ab) const;

  /** Methods to sort the LCTs */
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>&,
                int bx,
                bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const;
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>&,
                bool (*sorter)(const CSCCorrelatedLCTDigi&, const CSCCorrelatedLCTDigi&)) const;

  /* check if an ALCT cross a CLCT in an ME11 sector */
  bool doesALCTCrossCLCT(const CSCALCTDigi& a, const CSCCLCTDigi& c) const;

  /* does wiregroup cross halfstrip or not */
  bool doesWiregroupCrossStrip(int key_wg, int key_hs) const override;

  /* correlate a pair of ALCTs and a pair of CLCTs with matched pads or copads
     the output is up to two LCTs in a sector of ME11 */
  void correlateLCTsGEM(const CSCALCTDigi& bestALCT,
                        const CSCALCTDigi& secondALCT,
                        const CSCCLCTDigi& bestCLCT,
                        const CSCCLCTDigi& secondCLCT,
                        const GEMPadDigiIds& pads,
                        const GEMCoPadDigiIds& copads,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  /* store the CLCTs found separately in ME1a and ME1b */
  std::vector<CSCCLCTDigi> clctV;

  // Drop low quality stubs if they don't have GEMs
  bool dropLowQualityCLCTsNoGEMs_ME1a_;
  bool dropLowQualityCLCTsNoGEMs_ME1b_;
  bool dropLowQualityALCTsNoGEMs_ME1a_;
  bool dropLowQualityALCTsNoGEMs_ME1b_;

  // build LCT from ALCT and GEM
  bool buildLCTfromALCTandGEM_ME1a_;
  bool buildLCTfromALCTandGEM_ME1b_;
  bool buildLCTfromCLCTandGEM_ME1a_;
  bool buildLCTfromCLCTandGEM_ME1b_;

  // promote ALCT-GEM quality
  bool promoteCLCTGEMquality_ME1a_;
  bool promoteCLCTGEMquality_ME1b_;
};
#endif
