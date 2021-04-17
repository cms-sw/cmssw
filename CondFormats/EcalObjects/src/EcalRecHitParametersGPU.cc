#include "CondFormats/EcalObjects/interface/EcalRecHitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

EcalRecHitParametersGPU::EcalRecHitParametersGPU(edm::ParameterSet const& ps) {
  auto const& channelStatusToBeExcluded = StringToEnumValue<EcalChannelStatusCode::Code>(
      ps.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));

  channelStatusToBeExcluded_.resize(channelStatusToBeExcluded.size());
  std::copy(channelStatusToBeExcluded.begin(), channelStatusToBeExcluded.end(), channelStatusToBeExcluded_.begin());

  //     https://github.com/cms-sw/cmssw/blob/266e21cfc9eb409b093e4cf064f4c0a24c6ac293/RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.cc

  // Translate string representation of flagsMapDBReco into enum values
  const edm::ParameterSet& p = ps.getParameter<edm::ParameterSet>("flagsMapDBReco");
  std::vector<std::string> recoflagbitsStrings = p.getParameterNames();

  for (unsigned int i = 0; i != recoflagbitsStrings.size(); ++i) {
    EcalRecHit::Flags recoflagbit = (EcalRecHit::Flags)StringToEnumValue<EcalRecHit::Flags>(recoflagbitsStrings[i]);
    std::vector<std::string> dbstatus_s = p.getParameter<std::vector<std::string>>(recoflagbitsStrings[i]);
    for (unsigned int j = 0; j != dbstatus_s.size(); ++j) {
      EcalChannelStatusCode::Code dbstatus =
          (EcalChannelStatusCode::Code)StringToEnumValue<EcalChannelStatusCode::Code>(dbstatus_s[j]);
      expanded_v_DB_reco_flags_.push_back(dbstatus);
    }

    expanded_Sizes_v_DB_reco_flags_.push_back(dbstatus_s.size());
    expanded_flagbit_v_DB_reco_flags_.push_back(recoflagbit);
  }
}

EcalRecHitParametersGPU::Product const& EcalRecHitParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalRecHitParametersGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.channelStatusToBeExcluded =
            cms::cuda::make_device_unique<int[]>(channelStatusToBeExcluded_.size(), cudaStream);
        product.expanded_v_DB_reco_flags =
            cms::cuda::make_device_unique<int[]>(expanded_v_DB_reco_flags_.size(), cudaStream);
        product.expanded_Sizes_v_DB_reco_flags =
            cms::cuda::make_device_unique<uint32_t[]>(expanded_Sizes_v_DB_reco_flags_.size(), cudaStream);
        product.expanded_flagbit_v_DB_reco_flags =
            cms::cuda::make_device_unique<uint32_t[]>(expanded_flagbit_v_DB_reco_flags_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.channelStatusToBeExcluded, channelStatusToBeExcluded_, cudaStream);
        cms::cuda::copyAsync(product.expanded_v_DB_reco_flags, expanded_v_DB_reco_flags_, cudaStream);
        cms::cuda::copyAsync(product.expanded_Sizes_v_DB_reco_flags, expanded_Sizes_v_DB_reco_flags_, cudaStream);
        cms::cuda::copyAsync(product.expanded_flagbit_v_DB_reco_flags, expanded_flagbit_v_DB_reco_flags_, cudaStream);
      });
  return product;
}

TYPELOOKUP_DATA_REG(EcalRecHitParametersGPU);
