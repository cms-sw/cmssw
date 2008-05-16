#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvent.h"


// empty constructor
TtSemiEvent::TtSemiEvent():
  decay_(kNone), fitChi2_(-1.), genMatchSumPt_(-1.), genMatchSumDR_(-1.)
{
}

// // contructor from values
// TtSemiEvent::TtSemiEvent():
//   decay_(kNone), fitChi2_(-1.), genMatchSumPt_(-1.), genMatchSumDR_(-1.)
// {
// }
