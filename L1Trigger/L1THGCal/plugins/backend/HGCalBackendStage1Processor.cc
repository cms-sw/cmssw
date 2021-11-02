#include "L1Trigger/L1THGCal/interface/backend/HGCalBackendStage1Processor.h"

DEFINE_EDM_PLUGIN(HGCalBackendStage1Factory, HGCalBackendStage1Processor, "HGCalBackendStage1Processor");

HGCalBackendStage1Processor::HGCalBackendStage1Processor(const edm::ParameterSet& conf)
    : HGCalBackendStage1ProcessorBase(conf) {
  truncation_ = std::make_unique<HGCalStage1TruncationImpl>(conf.getParameterSet("truncation_parameters"));
}

void HGCalBackendStage1Processor::run(
    const std::pair<uint32_t, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>>& fpga_id_tcs,
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& truncated_tcs,
    const edm::EventSetup& es) {
  if (truncation_)
    truncation_->eventSetup(es);

  truncation_->run(fpga_id_tcs.first, fpga_id_tcs.second, truncated_tcs);
}
