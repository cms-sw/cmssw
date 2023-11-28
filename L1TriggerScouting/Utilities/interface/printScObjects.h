#ifndef L1TriggerScouting_Utilities_printScObjects_h
#define L1TriggerScouting_Utilities_printScObjects_h

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

namespace l1ScoutingRun3 {

  void printScMuon(const ScMuon& muon,  std::ostream &outs = std::cout);    
  template <typename T>
  void printScCaloObject(const T& obj, std::ostream &outs = std::cout);
  void printScJet(const ScJet& jet, std::ostream &outs = std::cout);
  void printScEGamma(const ScEGamma& eGamma, std::ostream &outs = std::cout);
  void printScTau(const ScTau& tau, std::ostream &outs = std::cout);
  void printScBxSums(const ScBxSums& sums, std::ostream &outs = std::cout);

} // end namespace L1ScoutingRun3

#endif // L1TriggerScouting_Utilities_printScObjects_h