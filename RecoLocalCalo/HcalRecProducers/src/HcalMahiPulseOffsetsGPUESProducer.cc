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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalMahiPulseOffsetsGPU.h"

class HcalMahiPulseOffsetsGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HcalMahiPulseOffsetsGPUESProducer(edm::ParameterSet const&);
  ~HcalMahiPulseOffsetsGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<HcalMahiPulseOffsetsGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  std::vector<int> pulseOffsets_;
};

HcalMahiPulseOffsetsGPUESProducer::HcalMahiPulseOffsetsGPUESProducer(edm::ParameterSet const& pset)
    : pulseOffsets_(pset.getParameter<std::vector<int>>("pulseOffsets")) {
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void HcalMahiPulseOffsetsGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void HcalMahiPulseOffsetsGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<HcalMahiPulseOffsetsGPU> HcalMahiPulseOffsetsGPUESProducer::produce(JobConfigurationGPURecord const&) {
  return std::make_unique<HcalMahiPulseOffsetsGPU>(pulseOffsets_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(HcalMahiPulseOffsetsGPUESProducer);
