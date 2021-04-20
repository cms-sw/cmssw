// Author: Felice Pantaleo,Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef __RecoHGCal_TICL_PRbyCA_H__
#define __RecoHGCal_TICL_PRbyCA_H__
#include <memory>  // unique_ptr
#include "RecoHGCal/TICL/interface/PatternRecognitionAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "HGCGraph.h"

namespace ticl {
  template <typename TILES>
  class PatternRecognitionbyCA final : public PatternRecognitionAlgoBaseT<TILES> {
  public:
    PatternRecognitionbyCA(const edm::ParameterSet& conf, const CacheBase* cache, edm::ConsumesCollector iC);
    ~PatternRecognitionbyCA() override;

    void makeTracksters(const typename PatternRecognitionAlgoBaseT<TILES>::Inputs& input,
                        std::vector<Trackster>& result,
                        std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) override;

    void energyRegressionAndID(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& result);
    void emptyTrackstersFromSeedsTRK(std::vector<Trackster>& tracksters,
                                     std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation,
                                     const edm::ProductID& collectionID) const;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<int>("algo_verbosity", 0);
      iDesc.add<bool>("oneTracksterPerTrackSeed", false);
      iDesc.add<bool>("promoteEmptyRegionToTrackster", false);
      iDesc.add<bool>("out_in_dfs", true);
      iDesc.add<int>("max_out_in_hops", 10);
      iDesc.add<double>("min_cos_theta", 0.915);
      iDesc.add<double>("min_cos_pointing", -1.);
      iDesc.add<double>("root_doublet_max_distance_from_seed_squared", 9999);
      iDesc.add<double>("etaLimitIncreaseWindow", 2.1);
      iDesc.add<int>("skip_layers", 0);
      iDesc.add<int>("max_missing_layers_in_trackster", 9999);
      iDesc.add<int>("shower_start_max_layer", 9999);  // make default such that no filtering is applied
      iDesc.add<int>("min_layers_per_trackster", 10);
      iDesc.add<std::vector<int>>("filter_on_categories", {0});
      iDesc.add<double>("pid_threshold", 0.);                    // make default such that no filtering is applied
      iDesc.add<double>("energy_em_over_total_threshold", -1.);  // make default such that no filtering is applied
      iDesc.add<double>("max_longitudinal_sigmaPCA", 9999);
      iDesc.add<double>("max_delta_time", 3.);  //nsigma
      iDesc.add<std::string>("eid_input_name", "input");
      iDesc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
      iDesc.add<std::string>("eid_output_name_id", "output/id_probabilities");
      iDesc.add<double>("eid_min_cluster_energy", 1.);
      iDesc.add<int>("eid_n_layers", 50);
      iDesc.add<int>("eid_n_clusters", 10);
    }

  private:
    void mergeTrackstersTRK(const std::vector<Trackster>&,
                            const std::vector<reco::CaloCluster>&,
                            std::vector<Trackster>&,
                            std::unordered_map<int, std::vector<int>>& seedToTracksterAssociation) const;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    const std::unique_ptr<HGCGraphT<TILES>> theGraph_;
    const bool oneTracksterPerTrackSeed_;
    const bool promoteEmptyRegionToTrackster_;
    const bool out_in_dfs_;
    const unsigned int max_out_in_hops_;
    const float min_cos_theta_;
    const float min_cos_pointing_;
    const float root_doublet_max_distance_from_seed_squared_;
    const float etaLimitIncreaseWindow_;
    const int skip_layers_;
    const int max_missing_layers_in_trackster_;
    bool check_missing_layers_ = false;
    const unsigned int shower_start_max_layer_;
    const unsigned int min_layers_per_trackster_;
    const std::vector<int> filter_on_categories_;
    const double pid_threshold_;
    const double energy_em_over_total_threshold_;
    const double max_longitudinal_sigmaPCA_;
    const int min_clusters_per_ntuplet_;
    const float max_delta_time_;
    const std::string eidInputName_;
    const std::string eidOutputNameEnergy_;
    const std::string eidOutputNameId_;
    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;

    hgcal::RecHitTools rhtools_;
    tensorflow::Session* eidSession_;

    static const int eidNFeatures_ = 3;
  };

}  // namespace ticl
#endif
