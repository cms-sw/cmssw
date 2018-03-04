#include "RecoEgamma/EgammaTools/interface/EgammaRandomSeeds.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

uint32_t egamma::getRandomSeedFromSC(const edm::Event& iEvent,const reco::SuperClusterRef scRef)
{
  std::seed_seq seeder = {int(iEvent.id().event()), int(iEvent.id().luminosityBlock()), int(iEvent.id().run()),
			  int(scRef->seed()->seed().rawId()),int(scRef->hitsAndFractions().size())};
  uint32_t seed = 0, tries = 10;
  do {
    seeder.generate(&seed,&seed+1); tries++;
  } while (seed == 0 && tries < 10);
  return seed ? seed : iEvent.id().event() + 10000*scRef.key();
}
