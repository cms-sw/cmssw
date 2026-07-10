// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 05/2024

#ifndef RecoHGCal_TICL_PatternRecognitionbyRecovery_h
#define RecoHGCal_TICL_PatternRecognitionbyRecovery_h
#include <memory>  // unique_ptr
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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
    void setGeometry(hgcal::RecHitTools const& rhtools) override;

  private:
    float z_limit_em_ = 0.f;
  };

}  // namespace ticl

#endif  // RecoHGCal_TICL_PatternRecognitionbyRecovery_h
