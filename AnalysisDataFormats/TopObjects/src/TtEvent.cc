#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include <cstring>

// find corresponding hypotheses based on JetLepComb
int TtEvent::correspondingHypo(const HypoClassKey& key1, const unsigned& hyp1, const HypoClassKey& key2) const {
  for (unsigned hyp2 = 0; hyp2 < this->numberOfAvailableHypos(key2); ++hyp2) {
    if (this->jetLeptonCombination(key1, hyp1) == this->jetLeptonCombination(key2, hyp2))
      return hyp2;
  }
  return -1;  // if no corresponding hypothesis was found
}

// return the corresponding enum value from a string
TtEvent::HypoClassKey TtEvent::hypoClassKeyFromString(const std::string& label) const {
  return (HypoClassKey)StringToEnumValue<HypoClassKey>(label);
}
