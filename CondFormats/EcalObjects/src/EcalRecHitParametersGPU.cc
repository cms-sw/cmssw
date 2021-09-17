#include "CondFormats/EcalObjects/interface/EcalRecHitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "CondFormats/EcalObjects/interface/EcalRechitChannelStatusGPU.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

EcalRecHitParametersGPU::EcalRecHitParametersGPU(std::vector<int> const& channelStatusToBeExcluded,
                                                 std::vector<DBStatus> const& flagsMapDBReco) {
  channelStatusToBeExcluded_.resize(channelStatusToBeExcluded.size());
  std::copy(channelStatusToBeExcluded.begin(), channelStatusToBeExcluded.end(), channelStatusToBeExcluded_.begin());

  for (auto const& flagInfo : flagsMapDBReco) {
    EcalRecHit::Flags recoflagbit = static_cast<EcalRecHit::Flags>(flagInfo.recoflagbit);
    for (auto v : flagInfo.dbstatus) {
      EcalChannelStatusCode::Code dbstatus = static_cast<EcalChannelStatusCode::Code>(v);
      expanded_v_DB_reco_flags_.push_back(dbstatus);
    }

    expanded_Sizes_v_DB_reco_flags_.push_back(flagInfo.dbstatus.size());
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
