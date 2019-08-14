// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyCA_H__
#define __RecoHGCal_TICL_PRbyCA_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class HGCGraph;

namespace ticl {
  class PatternRecognitionbyCA final : public PatternRecognitionAlgoBase {
  public:
    PatternRecognitionbyCA(const edm::ParameterSet& conf, const CacheBase* cache);
    ~PatternRecognitionbyCA() override;

    void makeTracksters(const edm::Event& ev,
                        const edm::EventSetup& es,
                        const std::vector<reco::CaloCluster>& layerClusters,
                        const std::vector<float>& mask,
                        const edm::ValueMap<float>& layerClustersTime,
                        const TICLLayerTiles& tiles,
                        std::vector<Trackster>& result) override;

    void energyRegressionAndID(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& result);

  private:
    hgcal::RecHitTools rhtools_;
    std::unique_ptr<HGCGraph> theGraph_;
    float min_cos_theta_;
    float min_cos_pointing_;
    int missing_layers_;
    int min_clusters_per_ntuplet_;
    float max_delta_time_;
    tensorflow::Session* eidSession_;
    std::string eidInputName_;
    std::string eidOutputNameEnergy_;
    std::string eidOutputNameId_;
    float eidMinClusterEnergy_;
    int eidNLayers_;
    int eidNClusters_;

    static const int eidNFeatures_ = 3;
  };
}  // namespace ticl
#endif
