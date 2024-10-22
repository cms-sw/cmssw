#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"

class HGCalBackendLayer1Processor2DClustering : public HGCalBackendLayer1ProcessorBase {
public:
  HGCalBackendLayer1Processor2DClustering(const edm::ParameterSet& conf) : HGCalBackendLayer1ProcessorBase(conf) {
    std::string typeCluster(conf.getParameterSet("C2d_parameters").getParameter<std::string>("clusterType"));
    if (typeCluster == "dRC2d") {
      clusteringAlgorithmType_ = dRC2d;
      clustering_ = std::make_unique<HGCalClusteringImpl>(conf.getParameterSet("C2d_parameters"));
    } else if (typeCluster == "NNC2d") {
      clusteringAlgorithmType_ = NNC2d;
      clustering_ = std::make_unique<HGCalClusteringImpl>(conf.getParameterSet("C2d_parameters"));
    } else if (typeCluster == "dRNNC2d") {
      clusteringAlgorithmType_ = dRNNC2d;
      clustering_ = std::make_unique<HGCalClusteringImpl>(conf.getParameterSet("C2d_parameters"));
    } else if (typeCluster == "dummyC2d") {
      clusteringAlgorithmType_ = dummyC2d;
      clusteringDummy_ = std::make_unique<HGCalClusteringDummyImpl>(conf.getParameterSet("C2d_parameters"));
    } else {
      throw cms::Exception("HGCTriggerParameterError") << "Unknown clustering type '" << typeCluster;
    }
  }

  void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle,
           l1t::HGCalClusterBxCollection& collCluster2D) override {
    if (clustering_)
      clustering_->setGeometry(geometry());
    if (clusteringDummy_)
      clusteringDummy_->setGeometry(geometry());

    /* create a persistent vector of pointers to the trigger-cells */
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
    for (unsigned i = 0; i < collHandle->size(); ++i) {
      edm::Ptr<l1t::HGCalTriggerCell> ptr(collHandle, i);
      triggerCellsPtrs.push_back(ptr);
    }

    std::sort(triggerCellsPtrs.begin(),
              triggerCellsPtrs.end(),
              [](const edm::Ptr<l1t::HGCalTriggerCell>& a, const edm::Ptr<l1t::HGCalTriggerCell>& b) -> bool {
                return a->mipPt() > b->mipPt();
              });

    /* call to C2d clustering */
    switch (clusteringAlgorithmType_) {
      case dRC2d:
        clustering_->clusterizeDR(triggerCellsPtrs, collCluster2D);
        break;
      case NNC2d:
        clustering_->clusterizeNN(triggerCellsPtrs, collCluster2D, *geometry());
        break;
      case dRNNC2d:
        clustering_->clusterizeDRNN(triggerCellsPtrs, collCluster2D, *geometry());
        break;
      case dummyC2d:
        clusteringDummy_->clusterizeDummy(triggerCellsPtrs, collCluster2D);
        break;
      default:
        // Should not happen, clustering type checked in constructor
        break;
    }
  }

private:
  enum ClusterType { dRC2d, NNC2d, dRNNC2d, dummyC2d };

  /* algorithms instances */
  std::unique_ptr<HGCalClusteringImpl> clustering_;
  std::unique_ptr<HGCalClusteringDummyImpl> clusteringDummy_;

  /* algorithm type */
  ClusterType clusteringAlgorithmType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer1Factory,
                  HGCalBackendLayer1Processor2DClustering,
                  "HGCalBackendLayer1Processor2DClustering");
