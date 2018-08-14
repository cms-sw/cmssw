#ifndef RecoEgamma_EgammaTools_EgammaRandomSeeds_h
#define RecoEgamma_EgammaTools_EgammaRandomSeeds_h

//author: Sam Harper (RAL)
//description: 
//  provides a seed for a random number generator based on object / event properties
//  the supercluster method is prefered as it will be the same for electrons & photons
//  the SC is based on seed ID and #crystals in SC, the #crystals in SC is for added
//  entropy although this means any re-recos will change the returned number as #crystals
//  will likely change with any re-reco while seed ID is fairly stable

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <random>

namespace egamma{

  uint32_t getRandomSeedFromSC(const edm::Event& iEvent,const reco::SuperClusterRef scRef);
  template<typename T>
  uint32_t getRandomSeedFromObj(const edm::Event& iEvent,const T& obj,size_t nrObjs,size_t objNr){
    std::seed_seq seeder = {int(iEvent.id().event()), int(iEvent.id().luminosityBlock()), int(iEvent.id().run()),
			    int(nrObjs),int(std::numeric_limits<int>::max()*obj.phi()/M_PI) & 0xFFF,int(objNr)};	   
    uint32_t seed = 0, tries = 10;
    do {
      seeder.generate(&seed,&seed+1); tries++;
    } while (seed == 0 && tries < 10);
    return seed ? seed : iEvent.id().event() + 10000*objNr;
  }
}

#endif
