#ifndef RecoLocalCalo_HcalRecAlgos_fetchHcalAlgoData_h_
#define RecoLocalCalo_HcalRecAlgos_fetchHcalAlgoData_h_

#include <memory>
#include <string>

#include "CondFormats/HcalObjects/interface/AbsHcalAlgoData.h"
#include "FWCore/Framework/interface/EventSetup.h"

//
// Factory function for fetching AbsHcalAlgoData descendants
// from the event setup
//
std::unique_ptr<AbsHcalAlgoData> fetchHcalAlgoData(const std::string& className, const edm::EventSetup& es);

#endif  // RecoLocalCalo_HcalRecAlgos_fetchHcalAlgoData_h_
