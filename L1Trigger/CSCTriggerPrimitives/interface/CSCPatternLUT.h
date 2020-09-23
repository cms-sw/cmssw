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
#include <DataFormats/L1TMuon/interface/CSCConstants.h>

class CSCPatternLUT {
public:
  static double get2007Position(int pattern);

private:
};

#endif
