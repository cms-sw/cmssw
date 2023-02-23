#include <string>
#include <vector>
#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

class PFClusteringParamsGPUESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  PFClusteringParamsGPUESSource(edm::ParameterSet const&);
  ~PFClusteringParamsGPUESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  std::unique_ptr<PFClusteringParamsGPU> produce(JobConfigurationGPURecord const&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  edm::ParameterSet const& pset_;
};

PFClusteringParamsGPUESSource::PFClusteringParamsGPUESSource(edm::ParameterSet const& pset) : pset_{pset} {
  // the fwk automatically uses "appendToDataLabel" as product label of the ESProduct
  setWhatProduced(this);
  findingRecord<JobConfigurationGPURecord>();
}

void PFClusteringParamsGPUESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                                                   const edm::IOVSyncValue& iTime,
                                                   edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

void PFClusteringParamsGPUESSource::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription psetDesc;
  psetDesc.add<std::string>("appendToDataLabel", "pfClusParamsOffline");
  PFClusteringParamsGPU::fillDescription(psetDesc);
  desc.addWithDefaultLabel(psetDesc);
}

std::unique_ptr<PFClusteringParamsGPU> PFClusteringParamsGPUESSource::produce(JobConfigurationGPURecord const&) {
  return std::make_unique<PFClusteringParamsGPU>(pset_);
}

#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(PFClusteringParamsGPUESSource);
