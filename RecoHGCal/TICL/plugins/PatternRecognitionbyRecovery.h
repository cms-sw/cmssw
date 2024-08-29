// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 05/2024

#ifndef __RecoHGCal_TICL_PatternRecognitionbyRecovery_H__
#define __RecoHGCal_TICL_PatternRecognitionbyRecovery_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {
  template <typename TILES>
  class PatternRecognitionbyRecovery final : public PatternRecognitionAlgoBaseT<TILES> {
  public:
    PatternRecognitionbyRecovery(const edm::ParameterSet& conf, edm::ConsumesCollector);
    ~PatternRecognitionbyRecovery() override = default;

    void makeTracksters(const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                        std::vector<Trackster>& result,
                        std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    void filter(std::vector<Trackster>& output,
                const std::vector<Trackster>& inTracksters,
                const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    hgcal::RecHitTools rhtools_;
  };

}  // namespace ticl

#endif  // __RecoHGCal_TICL_PatternRecognitionbyRecovery_H__
