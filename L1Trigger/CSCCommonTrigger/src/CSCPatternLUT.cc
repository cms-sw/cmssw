#include <L1Trigger/CSCCommonTrigger/interface/CSCPatternLUT.h>

double CSCPatternLUT::get2007Position(int pattern) {
  double PositionList[CSCConstants::NUM_CLCT_PATTERNS] = {
      0.0, 0.0, -0.60, 0.60, -0.64, 0.64, -0.23, 0.23, -0.21, 0.21, 0.0};  // offset in the strip number for each pattern

  return PositionList[pattern];
}
