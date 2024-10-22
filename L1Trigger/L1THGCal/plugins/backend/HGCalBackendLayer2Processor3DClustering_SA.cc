#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoSeedingImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterInterpreterBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalStage2ClusterDistribution.h"

#include <utility>

class HGCalBackendLayer2Processor3DClusteringSA : public HGCalBackendLayer2ProcessorBase {
public:
  HGCalBackendLayer2Processor3DClusteringSA(const edm::ParameterSet& conf)
      : HGCalBackendLayer2ProcessorBase(conf),
        distributor_(conf.getParameterSet("DistributionParameters")),
        conf_(conf) {
    multiclusteringHistoSeeding_ = std::make_unique<HGCalHistoSeedingImpl>(
        conf.getParameterSet("C3d_parameters").getParameterSet("histoMax_C3d_seeding_parameters"));

    const edm::ParameterSet& clusteringParamConfig =
        conf.getParameterSet("C3d_parameters").getParameterSet("histoMax_C3d_clustering_parameters");
    const edm::ParameterSet& sortingTruncationParamConfig =
        conf.getParameterSet("C3d_parameters").getParameterSet("histoMax_C3d_sorting_truncation_parameters");
    const std::string& clusteringAlgoWrapperName = clusteringParamConfig.getParameter<std::string>("AlgoName");
    const std::string& sortingTruncationAlgoWrapperName =
        sortingTruncationParamConfig.getParameter<std::string>("AlgoName");

    multiclusteringHistoClusteringWrapper_ = std::unique_ptr<HGCalHistoClusteringWrapperBase>{
        HGCalHistoClusteringWrapperBaseFactory::get()->create(clusteringAlgoWrapperName, clusteringParamConfig)};
    multiclusteringSortingTruncationWrapper_ =
        std::unique_ptr<HGCalStage2FilteringWrapperBase>{HGCalStage2FilteringWrapperBaseFactory::get()->create(
            sortingTruncationAlgoWrapperName, sortingTruncationParamConfig)};

    for (const auto& interpretationPset : conf.getParameter<std::vector<edm::ParameterSet>>("energy_interpretations")) {
      std::unique_ptr<HGCalTriggerClusterInterpreterBase> interpreter{
          HGCalTriggerClusterInterpreterFactory::get()->create(interpretationPset.getParameter<std::string>("type"))};
      interpreter->initialize(interpretationPset);
      energy_interpreters_.push_back(std::move(interpreter));
    }
  }

  void run(const edm::Handle<l1t::HGCalClusterBxCollection>& collHandle,
           std::pair<l1t::HGCalMulticlusterBxCollection, l1t::HGCalClusterBxCollection>& be_output) override {
    if (multiclusteringHistoSeeding_)
      multiclusteringHistoSeeding_->setGeometry(geometry());
    l1t::HGCalMulticlusterBxCollection& collCluster3D_sorted = be_output.first;
    l1t::HGCalClusterBxCollection& rejectedClusters = be_output.second;

    /* create a persistent vector of pointers to the trigger-cells */
    std::unordered_map<uint32_t, std::vector<edm::Ptr<l1t::HGCalCluster>>> tcs_per_fpga;

    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalCluster> tc_ptr(collHandle, i);
      uint32_t module = geometry()->getModuleFromTriggerCell(tc_ptr->detId());
      uint32_t stage1_fpga = geometry()->getStage1FpgaFromModule(module);
      HGCalTriggerGeometryBase::geom_set possible_stage2_fpgas = geometry()->getStage2FpgasFromStage1Fpga(stage1_fpga);

      HGCalTriggerGeometryBase::geom_set stage2_fpgas =
          distributor_.getStage2FPGAs(stage1_fpga, possible_stage2_fpgas, tc_ptr);

      for (auto& fpga : stage2_fpgas) {
        tcs_per_fpga[fpga].push_back(tc_ptr);
      }
    }

    // Configuration
    const std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&> configuration{geometry(), conf_};
    multiclusteringHistoClusteringWrapper_->configure(configuration);
    multiclusteringSortingTruncationWrapper_->configure(configuration);

    for (auto& fpga_tcs : tcs_per_fpga) {
      /* create a vector of seed positions and their energy*/
      std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

      /* call to multiclustering and compute shower shape*/
      multiclusteringHistoSeeding_->findHistoSeeds(fpga_tcs.second, seedPositionsEnergy);

      // Inputs
      std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>&, const std::vector<std::pair<GlobalPoint, double>>&>
          inputClustersAndSeeds{fpga_tcs.second, seedPositionsEnergy};
      // Outputs
      l1t::HGCalMulticlusterBxCollection collCluster3D_perFPGA;
      l1t::HGCalMulticlusterBxCollection collCluster3D_perFPGA_sorted;
      l1t::HGCalClusterBxCollection rejectedClusters_perFPGA;

      std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>
          outputMulticlustersAndRejectedClusters_perFPGA{collCluster3D_perFPGA, rejectedClusters_perFPGA};

      // Process
      multiclusteringHistoClusteringWrapper_->process(inputClustersAndSeeds,
                                                      outputMulticlustersAndRejectedClusters_perFPGA);

      multiclusteringSortingTruncationWrapper_->process(collCluster3D_perFPGA, collCluster3D_perFPGA_sorted);

      // Call all the energy interpretation modules on the cluster collection
      for (const auto& interpreter : energy_interpreters_) {
        interpreter->setGeometry(geometry());
        interpreter->interpret(collCluster3D_perFPGA_sorted);
      }

      for (const auto& collcluster : collCluster3D_perFPGA_sorted) {
        collCluster3D_sorted.push_back(0, collcluster);
      }
      for (const auto& rejectedcluster : rejectedClusters_perFPGA) {
        rejectedClusters.push_back(0, rejectedcluster);
      }
    }
  }

private:
  /* algorithms instances */
  std::unique_ptr<HGCalHistoSeedingImpl> multiclusteringHistoSeeding_;

  std::unique_ptr<HGCalHistoClusteringWrapperBase> multiclusteringHistoClusteringWrapper_;

  std::unique_ptr<HGCalStage2FilteringWrapperBase> multiclusteringSortingTruncationWrapper_;

  std::vector<std::unique_ptr<HGCalTriggerClusterInterpreterBase>> energy_interpreters_;

  HGCalStage2ClusterDistribution distributor_;
  const edm::ParameterSet conf_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer2Factory,
                  HGCalBackendLayer2Processor3DClusteringSA,
                  "HGCalBackendLayer2Processor3DClusteringSA");
