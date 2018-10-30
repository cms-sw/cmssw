#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboardME11_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboardME11_h

/** \class CSCMotherboardME11
 *
 * Extended CSCMotherboard for ME11 TMB upgrade
 * to handle ME1b and (primarily unganged) ME1a separately
 *
 * \author Vadim Khotilovich 12 May 2009
 *
 * Based on CSCMotherboard code
 *
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"

class CSCMotherboardME11 : public CSCUpgradeMotherboard
{
 public:
  /** Normal constructor. */
  CSCMotherboardME11(unsigned endcap, unsigned station, unsigned sector,
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCMotherboardME11();

  /** Default destructor. */
  ~CSCMotherboardME11() override;

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc,
	   const CSCComparatorDigiCollection* compdc);

  /** Returns vectors of found correlated LCTs in ME1a and ME1b, if any. */
  std::vector<CSCCorrelatedLCTDigi> getLCTs1a() const;
  std::vector<CSCCorrelatedLCTDigi> getLCTs1b() const;

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Set configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1a() const;
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1b() const;
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs(int me1ab) const;

 private:

  std::unique_ptr<CSCMotherboardLUTME11> cscTmbLUT_;

  /** labels for ME1a and ME1B */
  enum {ME1B = 1, ME1A=4};

  bool doesALCTCrossCLCT(const CSCALCTDigi &a, const CSCCLCTDigi &c) const;

  void correlateLCTsME11(const CSCALCTDigi& bestALCT,
                         const CSCALCTDigi& secondALCT,
                         const CSCCLCTDigi& bestCLCT,
                         const CSCCLCTDigi& secondCLCT,
                         CSCCorrelatedLCTDigi& lct1,
                         CSCCorrelatedLCTDigi& lct2) const;

  bool ignoreAlctCrossClct;
};
#endif
