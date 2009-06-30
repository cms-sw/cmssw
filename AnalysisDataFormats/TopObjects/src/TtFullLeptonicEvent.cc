#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

// print info via MessageLogger
void
TtFullLeptonicEvent::print()
{
  edm::LogInfo log("TtFullLeptonicEvent");

  log << "++++++++++++++++++++++++++++++++++++++++++++++ \n";

  // to do: create printout here!
  // have a look at TtSemiLeptonicEvent::print() for examples

  log << "++++++++++++++++++++++++++++++++++++++++++++++";  
}
