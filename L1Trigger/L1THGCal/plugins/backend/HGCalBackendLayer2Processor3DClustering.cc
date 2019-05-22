#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticlusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoSeedingImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringImpl.h"

class HGCalBackendLayer2Processor3DClustering : public HGCalBackendLayer2ProcessorBase {
public:
  HGCalBackendLayer2Processor3DClustering(const edm::ParameterSet& conf) : HGCalBackendLayer2ProcessorBase(conf) {
    std::string typeMulticluster(conf.getParameterSet("C3d_parameters").getParameter<std::string>("type_multicluster"));
    if (typeMulticluster == "dRC3d") {
      multiclusteringAlgoType_ = dRC3d;
      multiclustering_ = std::make_unique<HGCalMulticlusteringImpl>(conf.getParameterSet("C3d_parameters"));
    } else if (typeMulticluster == "DBSCANC3d") {
      multiclusteringAlgoType_ = DBSCANC3d;
      multiclustering_ = std::make_unique<HGCalMulticlusteringImpl>(conf.getParameterSet("C3d_parameters"));
    } else if (typeMulticluster.find("Histo") != std::string::npos) {
      multiclusteringAlgoType_ = HistoC3d;
      multiclusteringHistoSeeding_ = std::make_unique<HGCalHistoSeedingImpl>(conf.getParameterSet("C3d_parameters"));
      multiclusteringHistoClustering_ =
          std::make_unique<HGCalHistoClusteringImpl>(conf.getParameterSet("C3d_parameters"));
    } else {
      throw cms::Exception("HGCTriggerParameterError") << "Unknown Multiclustering type '" << typeMulticluster << "'";
    }
  }

  void run(const edm::Handle<l1t::HGCalClusterBxCollection>& collHandle,
           l1t::HGCalMulticlusterBxCollection& collCluster3D,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);
    if (multiclustering_)
      multiclustering_->eventSetup(es);
    if (multiclusteringHistoSeeding_)
      multiclusteringHistoSeeding_->eventSetup(es);
    if (multiclusteringHistoClustering_)
      multiclusteringHistoClustering_->eventSetup(es);

    /* create a persistent vector of pointers to the trigger-cells */
    std::vector<edm::Ptr<l1t::HGCalCluster>> clustersPtrs;
    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalCluster> ptr(collHandle, i);
      clustersPtrs.push_back(ptr);
    }

    /* create a vector of seed positions and their energy*/
    std::vector<std::pair<GlobalPoint, double>> seedPositionsEnergy;

    /* call to multiclustering and compute shower shape*/
    switch (multiclusteringAlgoType_) {
      case dRC3d:
        multiclustering_->clusterizeDR(clustersPtrs, collCluster3D, *triggerGeometry_);
        break;
      case DBSCANC3d:
        multiclustering_->clusterizeDBSCAN(clustersPtrs, collCluster3D, *triggerGeometry_);
        break;
      case HistoC3d:
        multiclusteringHistoSeeding_->findHistoSeeds(clustersPtrs, seedPositionsEnergy);
        multiclusteringHistoClustering_->clusterizeHisto(
            clustersPtrs, seedPositionsEnergy, *triggerGeometry_, collCluster3D);
        break;
      default:
        // Should not happen, clustering type checked in constructor
        break;
    }
  }

private:
  enum MulticlusterType { dRC3d, DBSCANC3d, HistoC3d };

  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  /* algorithms instances */
  std::unique_ptr<HGCalMulticlusteringImpl> multiclustering_;
  std::unique_ptr<HGCalHistoSeedingImpl> multiclusteringHistoSeeding_;
  std::unique_ptr<HGCalHistoClusteringImpl> multiclusteringHistoClustering_;

  /* algorithm type */
  MulticlusterType multiclusteringAlgoType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer2Factory,
                  HGCalBackendLayer2Processor3DClustering,
                  "HGCalBackendLayer2Processor3DClustering");
