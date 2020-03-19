#include "FlagsCleanerECAL.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

FlagsCleanerECAL::FlagsCleanerECAL(const edm::ParameterSet& conf) : RecHitTopologicalCleanerBase(conf) {
  const std::vector<std::string> flagnames = conf.getParameter<std::vector<std::string> >("RecHitFlagsToBeExcluded");
  v_chstatus_excl_ = StringToEnumValue<EcalRecHit::Flags>(flagnames);
}

void FlagsCleanerECAL::clean(const edm::Handle<reco::PFRecHitCollection>& input, std::vector<bool>& mask) {
  auto const& hits = *input;

  for (uint16_t idx = 0; idx < hits.size(); ++idx) {
    if (!mask[idx])
      continue;  // don't need to re-mask things :-)
    const reco::PFRecHit& rechit = hits[idx];
    if (checkFlags(rechit))
      mask[idx] = false;
  }
}

// returns true if one of the flags in the exclusion list is up
bool FlagsCleanerECAL::checkFlags(const reco::PFRecHit& hit) {
  for (auto flag : v_chstatus_excl_) {  // check if one of the flags is up
    if (hit.flags() & (0x1 << flag))
      return true;
  }
  return false;
}
