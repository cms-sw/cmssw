// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 04/2021

#ifndef __RecoHGCal_TICL_PRbyCLUE3D_H__
#define __RecoHGCal_TICL_PRbyCLUE3D_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {
  template <typename TILES>
  class PatternRecognitionbyCLUE3D final : public PatternRecognitionAlgoBaseT<TILES> {
  public:
    PatternRecognitionbyCLUE3D(const edm::ParameterSet& conf, edm::ConsumesCollector);
    ~PatternRecognitionbyCLUE3D() override = default;

    void makeTracksters(const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                        std::vector<Trackster>& result,
                        std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    void energyRegressionAndID(const std::vector<reco::CaloCluster>& layerClusters,
                               const tensorflow::Session*,
                               std::vector<Trackster>& result);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    struct ClustersOnLayer {
      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> z;
      std::vector<float> r_over_absz;
      std::vector<float> radius;
      std::vector<float> eta;
      std::vector<float> phi;
      std::vector<int> cells;
      std::vector<int> algoId;  // hgcal_em = 6, hgcal_had = 7, hgcal_scintillator = 8, hfnose = 9
      std::vector<uint8_t> isSilicon;

      std::vector<float> energy;
      std::vector<float> rho;
      std::vector<float> z_extension;

      std::vector<std::pair<float, int>> delta;
      std::vector<std::pair<int, int>> nearestHigher;
      std::vector<int> clusterIndex;
      std::vector<unsigned int> layerClusterOriginalIdx;
      std::vector<std::vector<std::pair<int, int>>> followers;
      std::vector<bool> isSeed;

      void clear() {
        x.clear();
        y.clear();
        z.clear();
        r_over_absz.clear();
        radius.clear();
        eta.clear();
        phi.clear();
        cells.clear();
        algoId.clear();
        isSilicon.clear();
        energy.clear();
        rho.clear();
        z_extension.clear();
        delta.clear();
        nearestHigher.clear();
        clusterIndex.clear();
        layerClusterOriginalIdx.clear();
        followers.clear();
        isSeed.clear();
      }

      void shrink_to_fit() {
        x.shrink_to_fit();
        y.shrink_to_fit();
        z.shrink_to_fit();
        r_over_absz.shrink_to_fit();
        radius.shrink_to_fit();
        eta.shrink_to_fit();
        phi.shrink_to_fit();
        cells.shrink_to_fit();
        algoId.shrink_to_fit();
        isSilicon.shrink_to_fit();
        energy.shrink_to_fit();
        rho.shrink_to_fit();
        z_extension.shrink_to_fit();
        delta.shrink_to_fit();
        nearestHigher.shrink_to_fit();
        clusterIndex.shrink_to_fit();
        layerClusterOriginalIdx.shrink_to_fit();
        followers.shrink_to_fit();
        isSeed.shrink_to_fit();
      }
    };

    void reset() {
      for (auto& c : clusters_) {
        c.clear();
        c.shrink_to_fit();
      }
    }
    void calculateLocalDensity(const TILES&, const int layerId, const std::vector<std::pair<int, int>>&);
    void calculateDistanceToHigher(const TILES&, const int layerId, const std::vector<std::pair<int, int>>&);
    int findAndAssignTracksters(const TILES&, const std::vector<std::pair<int, int>>&);
    void dumpClusters(const TILES& tiles,
                      const std::vector<std::pair<int, int>>& layerIdx2layerandSoa,
                      const int) const;
    void dumpTracksters(const std::vector<std::pair<int, int>>& layerIdx2layerandSoa,
                        const int,
                        const std::vector<Trackster>&) const;
    void dumpTiles(const TILES&) const;

    std::vector<ClustersOnLayer> clusters_;
    std::vector<float> layersPosZ_;
    std::vector<int> tracksterSeedAlgoId_;

    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    const std::vector<double> criticalDensity_;
    const std::vector<double> criticalSelfDensity_;
    const std::vector<int> densitySiblingLayers_;
    const std::vector<double> densityEtaPhiDistanceSqr_;
    const std::vector<double> densityXYDistanceSqr_;
    const std::vector<double> kernelDensityFactor_;
    const bool densityOnSameLayer_;
    const bool nearestHigherOnSameLayer_;
    const bool useAbsoluteProjectiveScale_;
    const bool useClusterDimensionXY_;
    const bool rescaleDensityByZ_;
    const std::vector<double> criticalEtaPhiDistance_;
    const std::vector<double> criticalXYDistance_;
    const std::vector<int> criticalZDistanceLyr_;
    const std::vector<double> outlierMultiplier_;
    const std::vector<int> minNumLayerCluster_;
    const bool doPidCut_;
    const float cutHadProb_;
    const std::vector<int> filter_on_categories_;
    const std::string eidInputName_;
    const std::string eidOutputNameEnergy_;
    const std::string eidOutputNameId_;
    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;
    const bool computeLocalTime_;
    const bool usePCACleaning_;

    hgcal::RecHitTools rhtools_;
    tensorflow::Session* eidSession_;

    static const int eidNFeatures_ = 3;
  };

}  // namespace ticl
#endif
