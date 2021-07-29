#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include <unordered_map>

class PassThruSeedFinder : public SeedFinderBase {
public:
  PassThruSeedFinder(const edm::ParameterSet& conf);
  PassThruSeedFinder(const PassThruSeedFinder&) = delete;
  PassThruSeedFinder& operator=(const PassThruSeedFinder&) = delete;

  void findSeeds(const edm::Handle<reco::PFRecHitCollection>& input,
                 const std::vector<bool>& mask,
                 std::vector<bool>& seedable) override;

private:
};

DEFINE_EDM_PLUGIN(SeedFinderFactory, PassThruSeedFinder, "PassThruSeedFinder");

PassThruSeedFinder::PassThruSeedFinder(const edm::ParameterSet& conf) : SeedFinderBase(conf) {}

// the starting state of seedable is all false!
void PassThruSeedFinder::findSeeds(const edm::Handle<reco::PFRecHitCollection>& input,
                                   const std::vector<bool>& mask,
                                   std::vector<bool>& seedable) {
  seedable = std::vector<bool>(input->size(), true);
}
