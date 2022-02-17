#include "L1Trigger/L1THGCal/interface/backend/HGCalBackendStage1Processor.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"

DEFINE_EDM_PLUGIN(HGCalBackendStage1Factory, HGCalBackendStage1Processor, "HGCalBackendStage1Processor");

HGCalBackendStage1Processor::HGCalBackendStage1Processor(const edm::ParameterSet& conf)
    : HGCalBackendStage1ProcessorBase(conf), conf_(conf) {
  const edm::ParameterSet& truncationParamConfig = conf.getParameterSet("truncation_parameters");
  const std::string& truncationWrapperName = truncationParamConfig.getParameter<std::string>("AlgoName");

  truncationWrapper_ = std::unique_ptr<HGCalStage1TruncationWrapperBase>{
      HGCalStage1TruncationWrapperBaseFactory::get()->create(truncationWrapperName, truncationParamConfig)};
}

void HGCalBackendStage1Processor::run(
    const std::pair<uint32_t, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>>& fpga_id_tcs,
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& truncated_tcs) {
  const unsigned sector120 = HGCalTriggerBackendDetId(fpga_id_tcs.first).sector();
  const uint32_t fpga_id = fpga_id_tcs.first;

  // Configuration
  const std::tuple<const HGCalTriggerGeometryBase* const, unsigned, uint32_t> configuration{
      geometry(), sector120, fpga_id};
  truncationWrapper_->configure(configuration);

  truncationWrapper_->process(fpga_id_tcs.second, truncated_tcs);
}
