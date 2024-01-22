#ifndef L1TriggerScouting_Utilities_printScObjects_h
#define L1TriggerScouting_Utilities_printScObjects_h

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include "iostream"

namespace l1ScoutingRun3 {

  void printMuon(const Muon& muon, std::ostream& outs = std::cout);
  template <typename T>
  void printCaloObject(const T& obj, std::ostream& outs = std::cout);
  void printJet(const Jet& jet, std::ostream& outs = std::cout);
  void printEGamma(const EGamma& eGamma, std::ostream& outs = std::cout);
  void printTau(const Tau& tau, std::ostream& outs = std::cout);
  void printBxSums(const BxSums& sums, std::ostream& outs = std::cout);

}  // namespace l1ScoutingRun3

#endif  // L1TriggerScouting_Utilities_printScObjects_h