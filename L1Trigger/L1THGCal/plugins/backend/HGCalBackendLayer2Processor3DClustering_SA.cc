#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoSeedingImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterInterpreterBase.h"

#include <utility>

class HGCalBackendLayer2Processor3DClusteringSA : public HGCalBackendLayer2ProcessorBase {
public:
  HGCalBackendLayer2Processor3DClusteringSA(const edm::ParameterSet& conf)
      : HGCalBackendLayer2ProcessorBase(conf), conf_(conf) {
    multiclusteringHistoSeeding_ = std::make_unique<HGCalHistoSeedingImpl>(
        conf.getParameterSet("C3d_parameters").getParameterSet("histoMax_C3d_seeding_parameters"));

    const edm::ParameterSet& clusteringParamConfig =
        conf.getParameterSet("C3d_parameters").getParameterSet("histoMax_C3d_clustering_parameters");
    const std::string& clusteringAlgoWrapperName = clusteringParamConfig.getParameter<std::string>("AlgoName");
    multiclusteringHistoClusteringWrapper_ = std::unique_ptr<HGCalHistoClusteringWrapperBase>{
        HGCalHistoClusteringWrapperBaseFactory::get()->create(clusteringAlgoWrapperName, clusteringParamConfig)};

    for (const auto& interpretationPset : conf.getParameter<std::vector<edm::ParameterSet>>("energy_interpretations")) {
      std::unique_ptr<HGCalTriggerClusterInterpreterBase> interpreter{
          HGCalTriggerClusterInterpreterFactory::get()->create(interpretationPset.getParameter<std::string>("type"))};
      interpreter->initialize(interpretationPset);
      energy_interpreters_.push_back(std::move(interpreter));
    }
  }

  void run(const edm::Handle<l1t::HGCalClusterBxCollection>& collHandle,
           std::pair<l1t::HGCalMulticlusterBxCollection, l1t::HGCalClusterBxCollection>& be_output,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);
    if (multiclusteringHistoSeeding_)
      multiclusteringHistoSeeding_->eventSetup(es);
    l1t::HGCalMulticlusterBxCollection& collCluster3D = be_output.first;
    l1t::HGCalClusterBxCollection& rejectedClusters = be_output.second;

    /* create a persistent vector of pointers to the trigger-cells */
    std::vector<edm::Ptr<l1t::HGCalCluster>> clustersPtrs;
    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalCluster> ptr(collHandle, i);
      clustersPtrs.push_back(ptr);
    }

    /* create a vector of seed positions and their energy*/
    std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

    /* call to multiclustering and compute shower shape*/
    multiclusteringHistoSeeding_->findHistoSeeds(clustersPtrs, seedPositionsEnergy);

    // Inputs
    std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>&, const std::vector<std::pair<GlobalPoint, double>>&>
        inputClustersAndSeeds{clustersPtrs, seedPositionsEnergy};
    // Outputs
    std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>
        outputMulticlustersAndRejectedClusters{collCluster3D, rejectedClusters};
    // Configuration
    const std::pair<const edm::EventSetup&, const edm::ParameterSet&> configuration{es, conf_};

    // Configure and process
    multiclusteringHistoClusteringWrapper_->configure(configuration);
    multiclusteringHistoClusteringWrapper_->process(inputClustersAndSeeds, outputMulticlustersAndRejectedClusters);

    // Call all the energy interpretation modules on the cluster collection
    for (const auto& interpreter : energy_interpreters_) {
      interpreter->eventSetup(es);
      interpreter->interpret(collCluster3D);
    }
  }

private:
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  /* algorithms instances */
  std::unique_ptr<HGCalHistoSeedingImpl> multiclusteringHistoSeeding_;

  std::unique_ptr<HGCalHistoClusteringWrapperBase> multiclusteringHistoClusteringWrapper_;

  std::vector<std::unique_ptr<HGCalTriggerClusterInterpreterBase>> energy_interpreters_;

  const edm::ParameterSet conf_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer2Factory,
                  HGCalBackendLayer2Processor3DClusteringSA,
                  "HGCalBackendLayer2Processor3DClusteringSA");
