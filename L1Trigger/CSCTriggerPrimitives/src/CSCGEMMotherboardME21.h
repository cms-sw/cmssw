#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboardME21_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboardME21_h

/** \class CSCGEMMotherboardME21
 *
 * Extended CSCMotherboard for ME21 TMB upgrade
 *
 * \author Sven Dildick March 2014
 *
 * Based on CSCMotherboard code
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboard.h"

class CSCGEMMotherboardME21 : public CSCGEMMotherboard
{
 public:

  /** Normal constructor. */
  CSCGEMMotherboardME21(unsigned endcap, unsigned station, unsigned sector,
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Test destructor. */
  CSCGEMMotherboardME21();

  /** Default destructor. */
  ~CSCGEMMotherboardME21() override;

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc,
           const GEMPadDigiCollection* gemPads) override;

  /* readout the two best LCTs in this CSC */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const;

 private:

  /* access to the LUTs needed for matching */
  const CSCGEMMotherboardLUTME21* getLUT() const override {return tmbLUT_.get();}
  std::unique_ptr<CSCGEMMotherboardLUTME21> tmbLUT_;

  /* correlate a pair of ALCTs and a pair of CLCTs with matched pads or copads
     the output is up to two LCTs in a sector of ME21 */
  void correlateLCTsGEM(const CSCALCTDigi& bestALCT,
                        const CSCALCTDigi& secondALCT,
                        const CSCCLCTDigi& bestCLCT,
                        const CSCCLCTDigi& secondCLCT,
                        const GEMPadDigiIds& pads,
                        const GEMCoPadDigiIds& copads,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2,
                        enum CSCPart p) const;

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  LCTContainer allLCTs;

  /* store the CLCTs found earlier */
  std::vector<CSCCLCTDigi> clctV;

  // drop low quality stubs if they don't have GEMs
  bool dropLowQualityCLCTsNoGEMs_;
  bool dropLowQualityALCTsNoGEMs_;

  // build LCT from ALCT and GEM
  bool buildLCTfromALCTandGEM_;
  bool buildLCTfromCLCTandGEM_;
};
#endif
