#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboardME3141_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboardME3141_h

#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboard.h"

class CSCMotherboardME3141 : public CSCUpgradeMotherboard
{
public:

  // standard constructor
  CSCMotherboardME3141(unsigned endcap, unsigned station, unsigned sector,
                    unsigned subsector, unsigned chamber,
                    const edm::ParameterSet& conf);

   //Default constructor for testing
  CSCMotherboardME3141();

  ~CSCMotherboardME3141() override;

  // run TMB with GEM pad clusters as input
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc);

  void correlateLCTs(const CSCALCTDigi& bestALCT, const CSCALCTDigi& secondALCT,
                     const CSCCLCTDigi& bestCLCT, const CSCCLCTDigi& secondCLCT,
                     CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2) const;

  /* readout the two best LCTs in this CSC */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const;
};

#endif
