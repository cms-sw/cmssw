// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 10/2021

#ifndef __RecoHGCal_TICL_PRbyFASTJET_H__
#define __RecoHGCal_TICL_PRbyFASTJET_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

// fwd declaration

namespace fastjet {
  class PseudoJet;
};

namespace ticl {
  template <typename TILES>
  class PatternRecognitionbyFastJet final : public PatternRecognitionAlgoBaseT<TILES> {
  public:
    PatternRecognitionbyFastJet(const edm::ParameterSet& conf, edm::ConsumesCollector);
    ~PatternRecognitionbyFastJet() override = default;

    void makeTracksters(const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                        std::vector<Trackster>& result,
                        std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    void filter(std::vector<Trackster>& output,
                const std::vector<Trackster>& inTracksters,
                const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    void energyRegressionAndID(const std::vector<reco::CaloCluster>& layerClusters,
                               const tensorflow::Session*,
                               std::vector<Trackster>& result);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    const double antikt_radius_;
    const int minNumLayerCluster_;
    const bool computeLocalTime_;

    hgcal::RecHitTools rhtools_;

    void buildJetAndTracksters(std::vector<fastjet::PseudoJet>&, std::vector<ticl::Trackster>&);
  };

}  // namespace ticl
#endif
