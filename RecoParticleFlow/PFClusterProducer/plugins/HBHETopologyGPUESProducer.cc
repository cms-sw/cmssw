#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPU.h"

class HBHETopologyGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HBHETopologyGPUESProducer(edm::ParameterSet const&);
  ~HBHETopologyGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<HBHETopologyGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
		      const edm::IOVSyncValue&,
		      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const& pset_;
};

HBHETopologyGPUESProducer::HBHETopologyGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void HBHETopologyGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void HBHETopologyGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<HBHETopologyGPU> HBHETopologyGPUESProducer::produce(JobConfigurationGPURecord const&) {
  return std::make_unique<HBHETopologyGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(HBHETopologyGPUESProducer);
