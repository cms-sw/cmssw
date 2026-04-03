
#include "RecoLocalCalo/HGCalRecProducers/plugins/alpaka/LayerClusterMergingAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  ALPAKA_FN_ACC void KernelMergeLayerClusters::operator()(const Acc1D& acc,
                                                          reco::CaloClusterDeviceCollection::View merged,
                                                          reco::CaloClusterDeviceCollection::ConstView input,
                                                          uint32_t start) const {
    for (auto idx : alpaka::uniformElements(acc, input.position().metadata().size())) {
      auto cumulative_index = idx + start;
      merged.position().x()[cumulative_index] = merged.position().x()[idx];
      merged.position().y()[cumulative_index] = merged.position().y()[idx];
      merged.position().z()[cumulative_index] = merged.position().z()[idx];
      merged.energy().energy()[cumulative_index] = merged.energy().energy()[idx];
      merged.energy().correctedEnergy()[cumulative_index] = merged.energy().correctedEnergy()[idx];
      merged.energy().correctedEnergyUncertainty()[cumulative_index] =
          merged.energy().correctedEnergyUncertainty()[idx];
      merged.indexes().caloID()[cumulative_index] = merged.indexes().caloID()[idx];
      merged.indexes().algoID()[cumulative_index] = merged.indexes().algoID()[idx];
      merged.indexes().seedID()[cumulative_index] = merged.indexes().seedID()[idx];
      merged.indexes().flags()[cumulative_index] = merged.indexes().flags()[idx];
    }
  }

  void LayerClusterMerger::merge(Queue& queue,
                                 reco::CaloClusterDeviceCollection::View merged,
                                 reco::CaloClusterDeviceCollection::ConstView input,
                                 uint32_t& start) {
    const auto blocksize = 1024u;
    const auto gridsize = cms::alpakatools::divide_up_by(input.position().metadata().size(), blocksize);
    const auto workdivision = cms::alpakatools::make_workdiv<Acc1D>(gridsize, blocksize);
    alpaka::exec<Acc1D>(queue, workdivision, KernelMergeLayerClusters{}, merged, input, start);
    start += input.position().metadata().size();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
