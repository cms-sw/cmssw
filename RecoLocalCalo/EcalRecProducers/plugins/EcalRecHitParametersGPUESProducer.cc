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
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitParametersGPU.h"

class EcalRecHitParametersGPUESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  EcalRecHitParametersGPUESProducer(edm::ParameterSet const&);
  ~EcalRecHitParametersGPUESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  std::unique_ptr<EcalRecHitParametersGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const pset_;
};

EcalRecHitParametersGPUESProducer::EcalRecHitParametersGPUESProducer(edm::ParameterSet const& pset) : pset_{pset} {
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void EcalRecHitParametersGPUESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                       const edm::IOVSyncValue& iTime,
                                                       edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void EcalRecHitParametersGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription d;

  //---- db statuses to be exluded from reconstruction
  d.add<std::vector<std::string>>("ChannelStatusToBeExcluded",
                                  {
                                      "kDAC",
                                      "kNoisy",
                                      "kNNoisy",
                                      "kFixedG6",
                                      "kFixedG1",
                                      "kFixedG0",
                                      "kNonRespondingIsolated",
                                      "kDeadVFE",
                                      "kDeadFE",
                                      "kNoDataNoTP",
                                  });

  // reco flags association to DB flag
  edm::ParameterSetDescription desc_list_flagsMapDBReco;
  desc_list_flagsMapDBReco.add<std::vector<std::string>>("kGood", {"kOk", "kDAC", "kNoLaser", "kNoisy"});
  desc_list_flagsMapDBReco.add<std::vector<std::string>>("kNoisy", {"kNNoisy", "kFixedG6", "kFixedG1"});
  desc_list_flagsMapDBReco.add<std::vector<std::string>>("kNeighboursRecovered",
                                                         {"kFixedG0", "kNonRespondingIsolated", "kDeadVFE"});
  desc_list_flagsMapDBReco.add<std::vector<std::string>>("kTowerRecovered", {"kDeadFE"});
  desc_list_flagsMapDBReco.add<std::vector<std::string>>("kDead", {"kNoDataNoTP"});

  d.add<edm::ParameterSetDescription>("flagsMapDBReco", desc_list_flagsMapDBReco);

  desc.addWithDefaultLabel(d);
}

std::unique_ptr<EcalRecHitParametersGPU> EcalRecHitParametersGPUESProducer::produce(JobConfigurationGPURecord const&) {
  return std::make_unique<EcalRecHitParametersGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(EcalRecHitParametersGPUESProducer);
