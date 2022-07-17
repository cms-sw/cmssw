#include "L1Trigger/L1THGCal/interface/backend/HGCalBackendLayer1Processor.h"

DEFINE_EDM_PLUGIN(HGCalBackendLayer1Factory, HGCalBackendLayer1Processor, "HGCalBackendLayer1Processor");

HGCalBackendLayer1Processor::HGCalBackendLayer1Processor(const edm::ParameterSet& conf)
    : HGCalBackendLayer1ProcessorBase(conf) {
  clusteringDummy_ = std::make_unique<HGCalClusteringDummyImpl>(conf.getParameterSet("C2d_parameters"));
  truncation_ = std::make_unique<HGCalStage1TruncationImpl>(conf.getParameterSet("truncation_parameters"));
}

void HGCalBackendLayer1Processor::run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& collHandle,
                                      l1t::HGCalClusterBxCollection& collCluster2D) {
  if (clusteringDummy_)
    clusteringDummy_->setGeometry(geometry());
  if (truncation_)
    truncation_->setGeometry(geometry());

  std::unordered_map<uint32_t, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>> tcs_per_fpga;

  for (unsigned i = 0; i < collHandle->size(); ++i) {
    edm::Ptr<l1t::HGCalTriggerCell> tc_ptr(collHandle, i);
    uint32_t module = geometry()->getModuleFromTriggerCell(tc_ptr->detId());
    uint32_t fpga = geometry()->getStage1FpgaFromModule(module);
    tcs_per_fpga[fpga].push_back(tc_ptr);
  }

  std::vector<edm::Ptr<l1t::HGCalTriggerCell>> truncated_tcs;
  for (auto& fpga_tcs : tcs_per_fpga) {
    truncation_->run(fpga_tcs.first, fpga_tcs.second, truncated_tcs);
  }
  clusteringDummy_->clusterizeDummy(truncated_tcs, collCluster2D);
}
