#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"

class FlagsCleanerECAL : public RecHitTopologicalCleanerBase {
public:
  FlagsCleanerECAL(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);
  FlagsCleanerECAL(const FlagsCleanerECAL&) = delete;
  FlagsCleanerECAL& operator=(const FlagsCleanerECAL&) = delete;

  // mark rechits which are flagged as one of the values provided in the vector
  void clean(const edm::Handle<reco::PFRecHitCollection>& input, std::vector<bool>& mask) override;

private:
  std::vector<int> v_chstatus_excl_;  // list of rechit status flags to be excluded from seeding
  bool checkFlags(const reco::PFRecHit& hit);
};

DEFINE_EDM_PLUGIN(RecHitTopologicalCleanerFactory, FlagsCleanerECAL, "FlagsCleanerECAL");

FlagsCleanerECAL::FlagsCleanerECAL(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : RecHitTopologicalCleanerBase(conf, cc) {
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
