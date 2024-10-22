#include <array>
#include <tuple>
#include <utility>

#include "CondFormats/EcalObjects/interface/EcalRecHitParametersGPU.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
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
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

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
  auto const& channelStatusToBeExcluded = StringToEnumValue<EcalChannelStatusCode::Code>(
      pset_.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));

  //     https://github.com/cms-sw/cmssw/blob/266e21cfc9eb409b093e4cf064f4c0a24c6ac293/RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.cc

  // Translate string representation of flagsMapDBReco into enum values
  const edm::ParameterSet& p = pset_.getParameter<edm::ParameterSet>("flagsMapDBReco");
  std::vector<std::string> recoflagbitsStrings = p.getParameterNames();

  std::vector<EcalRecHitParametersGPU::DBStatus> status;
  status.reserve(recoflagbitsStrings.size());
  for (auto const& recoflagbitsString : recoflagbitsStrings) {
    EcalRecHit::Flags recoflagbit = (EcalRecHit::Flags)StringToEnumValue<EcalRecHit::Flags>(recoflagbitsString);
    std::vector<std::string> dbstatus_s = p.getParameter<std::vector<std::string>>(recoflagbitsString);

    std::vector<uint32_t> db_reco_flags;
    db_reco_flags.reserve(dbstatus_s.size());
    for (auto const& dbstatusString : dbstatus_s) {
      EcalChannelStatusCode::Code dbstatus =
          (EcalChannelStatusCode::Code)StringToEnumValue<EcalChannelStatusCode::Code>(dbstatusString);
      db_reco_flags.push_back(dbstatus);
    }
    status.emplace_back(static_cast<int>(recoflagbit), db_reco_flags);
  }

  return std::make_unique<EcalRecHitParametersGPU>(channelStatusToBeExcluded, status);
}

DEFINE_FWK_EVENTSETUP_SOURCE(EcalRecHitParametersGPUESProducer);
