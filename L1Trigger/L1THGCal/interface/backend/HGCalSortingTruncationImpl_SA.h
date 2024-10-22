#ifndef __L1Trigger_L1THGCal_HGCalSortingTruncationImplSA_h__
#define __L1Trigger_L1THGCal_HGCalSortingTruncationImplSA_h__

#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalSortingTruncationConfig_SA.h"

#include <vector>
#include <algorithm>

class HGCalSortingTruncationImplSA {
public:
  HGCalSortingTruncationImplSA() = default;
  ~HGCalSortingTruncationImplSA() = default;

  void sortAndTruncate_SA(const std::vector<l1thgcfirmware::HGCalMulticluster>& inputMulticlusters,
                          std::vector<l1thgcfirmware::HGCalMulticluster>& outputMulticlusters,
                          const l1thgcfirmware::SortingTruncationAlgoConfig& configuration) const;
};

#endif
