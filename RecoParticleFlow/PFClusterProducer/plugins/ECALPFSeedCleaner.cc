#include "ECALPFSeedCleaner.h"

ECALPFSeedCleaner::ECALPFSeedCleaner(const edm::ParameterSet& conf) : RecHitTopologicalCleanerBase(conf) {}

void ECALPFSeedCleaner::update(const edm::EventSetup& iSetup) { iSetup.get<EcalPFSeedingThresholdsRcd>().get(ths_); }

void ECALPFSeedCleaner::clean(const edm::Handle<reco::PFRecHitCollection>& input, std::vector<bool>& mask) {
  //need to run over energy sorted rechits, as this is order used in seeding step
  // this can cause ambiguity, isn't it better to index by detid ?
  auto const& hits = *input;
  std::vector<unsigned> ordered_hits(hits.size());
  for (unsigned i = 0; i < hits.size(); ++i)
    ordered_hits[i] = i;

  std::sort(ordered_hits.begin(), ordered_hits.end(), [&](unsigned i, unsigned j) {
    return hits[i].energy() > hits[j].energy();
  });

  for (const auto& idx : ordered_hits) {
    if (!mask[idx])
      continue;  // is it useful ?
    const reco::PFRecHit& rechit = hits[idx];

    float threshold = (*ths_)[rechit.detId()];
    if (rechit.energy() < threshold)
      mask[idx] = false;

  }  // rechit loop
}
