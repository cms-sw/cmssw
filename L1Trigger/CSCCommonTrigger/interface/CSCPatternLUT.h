#ifndef L1Trigger_CSCPatternLUT_h
#define L1Trigger_CSCPatternLUT_h

/**
 *\class CSCPatternLUT
 *\author L. Gray (UF)
 *
 * This class is a static interface to the CLCT Pattern LUT.
 * This was factored out of the Sector Receiver since it is used in 
 * parts of the trigger primitive generator (I think).
 */
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

class CSCPatternLUT {
public:
  static int getBendValue(int pattern);
  static double getPosition(int pattern);

  static double get2007Position(int pattern);

private:
  /** 
   * Fill the pattern lookup table. This table holds the average position
   * and bend for each pattern. The position is used to further improve
   * the phi resolution, and the bend is passed on to the track finding code
   * to allow it to better determine the tracks.
   * These were determined from Monte Carlo by running 100,000 events through
   * the code and finding the offset for each pattern type.
   * Note that the positions are unitless-- they are in "pattern widths"
   * meaning that they are in 1/2 strips for high pt patterns and distrips
   * for low pt patterns. BHT 26 June 2001
   */
};

#endif
