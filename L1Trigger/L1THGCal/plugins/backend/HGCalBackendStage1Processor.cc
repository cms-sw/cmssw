#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"

class HGCalBackendStage1Processor : public HGCalBackendLayer1ProcessorBase {
public:
  HGCalBackendStage1Processor(const edm::ParameterSet& conf) : HGCalBackendLayer1ProcessorBase(conf) {
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
           l1t::HGCalClusterBxCollection& collCluster2D,
           const edm::EventSetup& es) override {
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);
    if (clustering_)
      clustering_->eventSetup(es);
    if (clusteringDummy_)
      clusteringDummy_->eventSetup(es);
   const l1t::HGCalTriggerCellBxCollection& collInput = *collHandle; 
    /*To split jobs in fpga modules*/
    std::unordered_map<uint32_t,std::vector<l1t::HGCalTriggerCell>> triggerCellsPtrs;
//    std::vector<edm::Ptr<l1t::HGCalTriggerCell> > triggerCellsPtrs;
    for (const auto& trigMod : collInput) {
      uint32_t module = geometry_->getModuleFromTriggerCell(trigMod.detId());
      uint32_t fpga = geometry_->getStage1FpgaFromModule(trigMod.module());
      triggerCellsPtrs[module].push_back(trigMod);
    }

    /* create a persistent vector of pointers to the trigger-cells */
//    std::vector<edm::Ptr<l1t::HGCalTriggerCell>> triggerCellsPtrs;
//    for (unsigned i = 0; i < collHandle->size(); ++i) {
//      edm::Ptr<l1t::HGCalTriggerCell> ptr(collHandle, i);
//      triggerCellsPtrs.push_back(ptr);
//    }
     
    
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
        clustering_->clusterizeNN(triggerCellsPtrs, collCluster2D, *triggerGeometry_);
        break;
      case dRNNC2d:
        clustering_->clusterizeDRNN(triggerCellsPtrs, collCluster2D, *triggerGeometry_);
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

  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  /* algorithms instances */
  std::unique_ptr<HGCalClusteringImpl> clustering_;
  std::unique_ptr<HGCalClusteringDummyImpl> clusteringDummy_;

  /* algorithm type */
  ClusterType clusteringAlgorithmType_;
};

DEFINE_EDM_PLUGIN(HGCalBackendLayer1Factory,
                  HGCalBackendStage1Processor,
                  "HGCalBackendStage1Processor");
