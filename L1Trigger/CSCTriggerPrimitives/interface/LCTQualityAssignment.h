#ifndef L1Trigger_CSCTriggerPrimitives_LCTQualityAssignment_h
#define L1Trigger_CSCTriggerPrimitives_LCTQualityAssignment_h

/** \class LCTQualityAssignment
 *
 * Helper class to calculate the quality of an LCT. There
 * is a Run-2 quality (also used for Run-1) based on the
 * pattern Id and the number of layers. There are two Run-3
 * LCTs qualities. One for the non-GEM TMBs and one for OTMBs
 * which receive GEM information.
 *
 * \author Sven Dildick (Rice University)
 * \updates by Giovanni Mocellin (UC Davis)
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"

class LCTQualityAssignment : public CSCBaseboard {
public:
  // 4-bit LCT quality number. Made by TMB lookup tables and used for MPC sorting.
  enum class LCT_QualityRun2 : unsigned int {
    INVALID = 0,
    NO_CLCT = 1,
    NO_ALCT = 2,
    CLCT_LAYER_TRIGGER = 3,
    LOW_QUALITY = 4,
    MARGINAL_ANODE_CATHODE = 5,
    HQ_ANODE_MARGINAL_CATHODE = 6,
    HQ_CATHODE_MARGINAL_ANODE = 7,
    HQ_ACCEL_ALCT = 8,
    HQ_RESERVED_1 = 9,
    HQ_RESERVED_2 = 10,
    HQ_PATTERN_2_3 = 11,
    HQ_PATTERN_4_5 = 12,
    HQ_PATTERN_6_7 = 13,
    HQ_PATTERN_8_9 = 14,
    HQ_PATTERN_10 = 15
  };

  // See DN-20-016
  enum class LCT_QualityRun3 : unsigned int { INVALID = 0, LowQ = 1, MedQ = 2, HighQ = 3 };

  // See DN-20-016
  enum class LCT_QualityRun3GEM : unsigned int {
    INVALID = 0,
    CLCT_2GEM = 1,
    ALCT_2GEM = 2,
    ALCT_CLCT = 3,
    ALCT_CLCT_1GEM_CSCBend = 4,
    ALCT_CLCT_1GEM_GEMCSCBend = 5,
    ALCT_CLCT_2GEM_CSCBend = 6,
    ALCT_CLCT_2GEM_GEMCSCBend = 7
  };

  // constructor
  LCTQualityAssignment(unsigned endcap,
                       unsigned station,
                       unsigned sector,
                       unsigned subsector,
                       unsigned chamber,
                       CSCBaseboard::Parameters& conf);

  /** Default destructor. */
  ~LCTQualityAssignment() override {}

  // setter
  void setGEMCSCBending(const bool setBend) { assignGEMCSCBending_ = setBend; }

  // quality for all LCTs in Run-1/2 or Run-3
  unsigned findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT) const;
  unsigned findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, const GEMInternalCluster& cl) const;
  unsigned findQuality(const CSCALCTDigi& aLCT, const GEMInternalCluster& cl) const;
  unsigned findQuality(const CSCCLCTDigi& cLCT, const GEMInternalCluster& cl) const;

private:
  // quality for all LCTs in Run-1 and Run-2 (run-3 mode off)
  unsigned findQualityRun2(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT) const;

  // quality for non-ME1/1 LCTs in Run-3 without GEMs (run-3 mode on)
  unsigned findQualityRun3(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT) const;

  // quality for LCTs in Run-3 with GEMs (CCLUT off)
  // old-style to be compatible with EMTF Run-2
  unsigned findQualityGEMv1(const CSCCLCTDigi&, const GEMInternalCluster& cl) const;
  unsigned findQualityGEMv1(const CSCALCTDigi&, const CSCCLCTDigi&, const GEMInternalCluster& cl) const;

  // quality for LCTs in Run-3 with GEMs (CCLUT on)
  unsigned findQualityGEMv2(const CSCALCTDigi&, const CSCCLCTDigi&, const GEMInternalCluster& cl) const;

  bool assignGEMCSCBending_;
  bool runILT_;
};

#endif
