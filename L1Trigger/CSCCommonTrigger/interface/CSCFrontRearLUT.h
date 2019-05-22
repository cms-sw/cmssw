#ifndef L1Trigger_CSCFrontRearLUT_h
#define L1Trigger_CSCFrontRearLUT_h

/**
 * \class CSCFrontRearLUT
 * \author L.Gray
 * 
 * Ported from ORCA, factored out of CSCSectorReceiverLUT
 */

class CSCFrontRearLUT {
public:
  /**
   * This is a function which uses the variables to return the front/rear bit.
   * The calculation is done by considering how the chambers overlap each other.
   */
  static unsigned getFRBit(int sector, int subsector, int station, int cscid);
};

#endif
