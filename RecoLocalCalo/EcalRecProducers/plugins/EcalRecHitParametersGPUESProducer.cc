#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/typelookup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitParametersGPU.h"
#include "EcalRecHitParametersGPURecord.h"

#include "FWCore/Framework/interface/SourceFactory.h"

class EcalRecHitParametersGPUESProducer
        : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
    EcalRecHitParametersGPUESProducer(edm::ParameterSet const&);
    ~EcalRecHitParametersGPUESProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions&);
    std::unique_ptr<EcalRecHitParametersGPU> produce(EcalRecHitParametersGPURecord const&);

protected:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                        const edm::IOVSyncValue&,
                        edm::ValidityInterval&) override;

private:
    edm::ParameterSet const& pset_;
};

EcalRecHitParametersGPUESProducer::EcalRecHitParametersGPUESProducer(
        edm::ParameterSet const& pset) : pset_{pset}
{
    setWhatProduced(this);
    findingRecord<EcalRecHitParametersGPURecord>();
}

void EcalRecHitParametersGPUESProducer::setIntervalFor(
        const edm::eventsetup::EventSetupRecordKey& iKey,
        const edm::IOVSyncValue& iTime,
        edm::ValidityInterval& oInterval) {
    oInterval = edm::ValidityInterval(
        edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void EcalRecHitParametersGPUESProducer::fillDescriptions(
        edm::ConfigurationDescriptions& desc) {
    edm::ParameterSetDescription d;

    // ## db statuses to be exluded from reconstruction (some will be recovered)
    edm::ParameterSetDescription desc_ChannelStatusToBeExcluded;
    desc_ChannelStatusToBeExcluded.add<std::string>("kDAC");
    desc_ChannelStatusToBeExcluded.add<std::string>("kNoisy");
    desc_ChannelStatusToBeExcluded.add<std::string>("kNNoisy");
    desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG6");
    desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG1");
    desc_ChannelStatusToBeExcluded.add<std::string>("kFixedG0");
    desc_ChannelStatusToBeExcluded.add<std::string>("kNonRespondingIsolated");
    desc_ChannelStatusToBeExcluded.add<std::string>("kDeadVFE");
    desc_ChannelStatusToBeExcluded.add<std::string>("kDeadFE");
    desc_ChannelStatusToBeExcluded.add<std::string>("kNoDataNoTP");
    std::vector<edm::ParameterSet> default_ChannelStatusToBeExcluded(1);
    d.addVPSet("ChannelStatusToBeExcluded", desc_ChannelStatusToBeExcluded, default_ChannelStatusToBeExcluded);

    desc.addWithDefaultLabel(d);
}

std::unique_ptr<EcalRecHitParametersGPU> EcalRecHitParametersGPUESProducer::produce(
        EcalRecHitParametersGPURecord const&) {
    return std::make_unique<EcalRecHitParametersGPU>(pset_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(EcalRecHitParametersGPUESProducer);
