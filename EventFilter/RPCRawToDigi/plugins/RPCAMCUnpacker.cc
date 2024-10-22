#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpacker.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

RPCAMCUnpacker::RPCAMCUnpacker(edm::ParameterSet const&, edm::ConsumesCollector, edm::ProducesCollector) {}
RPCAMCUnpacker::~RPCAMCUnpacker() {}

void RPCAMCUnpacker::fillDescription(edm::ParameterSetDescription& desc) {
  edm::ParameterSetDescription pset;
  pset.add<bool>("fillAMCCounters", true);
  pset.add<int>("bxMin", -2);
  pset.add<int>("bxMax", +2);
  pset.add<int>("cppfDaqDelay", 0);
  desc.add<edm::ParameterSetDescription>("RPCAMCUnpackerSettings", pset);
}

void RPCAMCUnpacker::beginRun(edm::Run const& run, edm::EventSetup const& setup) {}

void RPCAMCUnpacker::produce(edm::Event& event,
                             edm::EventSetup const& setup,
                             std::map<RPCAMCLink, rpcamc13::AMCPayload> const& amc_payload) {}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpackerFactory.h"
DEFINE_EDM_PLUGIN(RPCAMCUnpackerFactory, RPCAMCUnpacker, "RPCAMCUnpacker");
