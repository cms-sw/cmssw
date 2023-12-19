#ifndef L1TriggerScouting_Utilities_convertToL1TFormat_h
#define L1TriggerScouting_Utilities_convertToL1TFormat_h

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include "iostream"

namespace l1ScoutingRun3 {

  l1t::Muon getL1TMuon(const Muon& muon);
  l1t::Jet getL1TJet(const Jet& jet);
  l1t::EGamma getL1TEGamma(const EGamma& eGamma);
  l1t::Tau getL1TTau(const Tau& scTau);
  l1t::EtSum getL1TEtSum(const BxSums& sums, l1t::EtSum::EtSumType);

}  // namespace l1ScoutingRun3

#endif  // L1TriggerScouting_Utilities_convertToL1TFormat_h