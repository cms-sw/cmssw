#ifndef FWCore_Utilities_BranchType_h
#define FWCore_Utilities_BranchType_h
/*----------------------------------------------------------------------

BranchType: The type of a Branch (Event, LuminosityBlock, Run, or ProcessBlock)

----------------------------------------------------------------------*/

namespace edm {
  // Note: These enum values are used as subscripts for a fixed size array, so they must not change.
  enum BranchType { InEvent = 0, InLumi = 1, InRun = 2, InProcess = 3, NumBranchTypes };
}  // namespace edm
#endif
