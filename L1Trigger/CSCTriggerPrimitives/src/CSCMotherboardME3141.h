#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboardME3141_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboardME3141_h

#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"

class CSCMotherboardME3141 : public CSCUpgradeMotherboard
{
public:

  enum Default_values{DEFAULT_MATCHING_VALUE = -99};

  // standard constructor
  CSCMotherboardME3141(unsigned endcap, unsigned station, unsigned sector,
                    unsigned subsector, unsigned chamber,
                    const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCMotherboardME3141();

  ~CSCMotherboardME3141() override;

  // clear stored pads and copads
  void clear();

  // run TMB with GEM pad clusters as input
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc);

  void correlateLCTs(CSCALCTDigi& bestALCT, CSCALCTDigi& secondALCT,
                     CSCCLCTDigi& bestCLCT, CSCCLCTDigi& secondCLCT,
                     CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2);

  /* readout the two best LCTs in this CSC */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const;

  /* store the CLCTs found earlier */
  std::vector<CSCCLCTDigi> clctV;

 private:
  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  LCTContainer allLCTs;
};

#endif
