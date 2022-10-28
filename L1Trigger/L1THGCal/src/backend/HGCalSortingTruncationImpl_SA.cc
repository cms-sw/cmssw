#include "L1Trigger/L1THGCal/interface/backend/HGCalSortingTruncationImpl_SA.h"

void HGCalSortingTruncationImplSA::sortAndTruncate_SA(
    const std::vector<l1thgcfirmware::HGCalMulticluster>& inputMulticlusters,
    std::vector<l1thgcfirmware::HGCalMulticluster>& outputMulticlusters,
    const l1thgcfirmware::SortingTruncationAlgoConfig& configuration) const {
  outputMulticlusters.reserve(inputMulticlusters.size());
  for (const auto& multicluster : inputMulticlusters) {
    outputMulticlusters.push_back(multicluster);
  }

  //Sort based on 3D cluster sum pT
  std::sort(outputMulticlusters.begin(),
            outputMulticlusters.end(),
            [](l1thgcfirmware::HGCalMulticluster& one, l1thgcfirmware::HGCalMulticluster& two) {
              return one.sumPt() < two.sumPt();
            });

  //Truncate, keeping maxTCs entries
  unsigned maxTCs = configuration.maxTCs();
  if (outputMulticlusters.size() > maxTCs) {
    outputMulticlusters.resize(maxTCs);
  }
}
