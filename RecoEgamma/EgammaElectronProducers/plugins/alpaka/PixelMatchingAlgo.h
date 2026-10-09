#ifndef RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h
#define RecoEgamma_EgammaElectronProducers_plugins_alpaka_PixelMatchingAlgo_h

#include "DataFormats/EgammaReco/interface/alpaka/ElectronSeedDeviceCollection.h"
#include "DataFormats/EgammaReco/interface/alpaka/SuperClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelMatchingAlgo {
  public:
    void printEleSeeds(Queue& queue, const reco::ElectronSeedDeviceCollection& collection) const;
    void printSCs(Queue& queue, const reco::SuperClusterDeviceCollection& collection) const;
    void matchSeeds(Queue& queue,
                    reco::ElectronSeedDeviceCollection& collection,
                    reco::SuperClusterDeviceCollection& collectionSCs,
                    double vtx_X,
                    double vtx_Y,
                    double vtx_Z) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
