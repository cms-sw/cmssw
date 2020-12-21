#include <array>
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
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalMultifitParametersGPU.h"

class EcalMultifitParametersGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  EcalMultifitParametersGPUESProducer(edm::ParameterSet const&);
  ~EcalMultifitParametersGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<EcalMultifitParametersGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const pset_;
};

EcalMultifitParametersGPUESProducer::EcalMultifitParametersGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void EcalMultifitParametersGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                         const edm::IOVSyncValue& iTime,
                                                         edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void EcalMultifitParametersGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<int>>("pulseOffsets", {-3, -2, -1, 0, 1, 2, 3, 4});
  d.add<std::vector<double>>("EBtimeFitParameters",
                             {-2.015452e+00,
                              3.130702e+00,
                              -1.234730e+01,
                              4.188921e+01,
                              -8.283944e+01,
                              9.101147e+01,
                              -5.035761e+01,
                              1.105621e+01});
  d.add<std::vector<double>>("EEtimeFitParameters",
                             {-2.390548e+00,
                              3.553628e+00,
                              -1.762341e+01,
                              6.767538e+01,
                              -1.332130e+02,
                              1.407432e+02,
                              -7.541106e+01,
                              1.620277e+01});
  d.add<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652});
  d.add<std::vector<double>>("EEamplitudeFitParameters", {1.890, 1.400});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<EcalMultifitParametersGPU> EcalMultifitParametersGPUESProducer::produce(
    JobConfigurationGPURecord const&) {
  return std::make_unique<EcalMultifitParametersGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(EcalMultifitParametersGPUESProducer);
