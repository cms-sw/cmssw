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
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"

class PFHBHERecHitParamsGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PFHBHERecHitParamsGPUESProducer(edm::ParameterSet const&);
  ~PFHBHERecHitParamsGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<PFHBHERecHitParamsGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
		      const edm::IOVSyncValue&,
		      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const& pset_;
};

PFHBHERecHitParamsGPUESProducer::PFHBHERecHitParamsGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void PFHBHERecHitParamsGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void PFHBHERecHitParamsGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;
  d.add<std::vector<int>>("depthHB", { 1, 2, 3, 4});
  d.add<std::vector<int>>("depthHE", { 1, 2, 3, 4, 5, 6, 7});
  d.add<std::vector<double>>("thresholdE_HB", { 0.1, 0.2, 0.3, 0.3});
  d.add<std::vector<double>>("thresholdE_HE", { 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
  desc.addWithDefaultLabel(d);
}

std::unique_ptr<PFHBHERecHitParamsGPU> PFHBHERecHitParamsGPUESProducer::produce(JobConfigurationGPURecord const&) {
  return std::make_unique<PFHBHERecHitParamsGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(PFHBHERecHitParamsGPUESProducer);
