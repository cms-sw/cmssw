#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRechitChannelStatusGPU.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

EcalRecHitParametersGPU::EcalRecHitParametersGPU(edm::ParameterSet const& ps) {
  auto const& ChannelStatusToBeExcluded = StringToEnumValue<EcalChannelStatusCode::Code>(
      ps.getParameter<std::vector<std::string>>("ChannelStatusToBeExcluded"));

  ChannelStatusToBeExcluded_.resize(ChannelStatusToBeExcluded.size());
  std::copy(ChannelStatusToBeExcluded.begin(), ChannelStatusToBeExcluded.end(), ChannelStatusToBeExcluded_.begin());

  //     https://github.com/cms-sw/cmssw/blob/266e21cfc9eb409b093e4cf064f4c0a24c6ac293/RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.cc

  // Traslate string representation of flagsMapDBReco into enum values
  const edm::ParameterSet& p = ps.getParameter<edm::ParameterSet>("flagsMapDBReco");
  std::vector<std::string> recoflagbitsStrings = p.getParameterNames();

  for (unsigned int i = 0; i != recoflagbitsStrings.size(); ++i) {
    EcalRecHit::Flags recoflagbit = (EcalRecHit::Flags)StringToEnumValue<EcalRecHit::Flags>(recoflagbitsStrings[i]);
    std::vector<std::string> dbstatus_s = p.getParameter<std::vector<std::string>>(recoflagbitsStrings[i]);
    //     std::vector<uint32_t> dbstatuses;
    for (unsigned int j = 0; j != dbstatus_s.size(); ++j) {
      EcalChannelStatusCode::Code dbstatus =
          (EcalChannelStatusCode::Code)StringToEnumValue<EcalChannelStatusCode::Code>(dbstatus_s[j]);
      expanded_v_DB_reco_flags_.push_back(dbstatus);
    }

    expanded_Sizes_v_DB_reco_flags_.push_back(dbstatus_s.size());
    expanded_flagbit_v_DB_reco_flags_.push_back(recoflagbit);
  }
}

EcalRecHitParametersGPU::Product::~Product() {
  cudaCheck(cudaFree(ChannelStatusToBeExcluded));
  cudaCheck(cudaFree(expanded_v_DB_reco_flags));
  cudaCheck(cudaFree(expanded_Sizes_v_DB_reco_flags));
  cudaCheck(cudaFree(expanded_flagbit_v_DB_reco_flags));
}

EcalRecHitParametersGPU::Product const& EcalRecHitParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalRecHitParametersGPU::Product& product, cudaStream_t cudaStream) {
        // malloc
        cudaCheck(cudaMalloc((void**)&product.ChannelStatusToBeExcluded,
                             this->ChannelStatusToBeExcluded_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.expanded_v_DB_reco_flags,
                             this->expanded_v_DB_reco_flags_.size() * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&product.expanded_Sizes_v_DB_reco_flags,
                             this->expanded_Sizes_v_DB_reco_flags_.size() * sizeof(uint32_t)));
        cudaCheck(cudaMalloc((void**)&product.expanded_flagbit_v_DB_reco_flags,
                             this->expanded_flagbit_v_DB_reco_flags_.size() * sizeof(uint32_t)));

        // transfer
        cudaCheck(cudaMemcpyAsync(product.ChannelStatusToBeExcluded,
                                  this->ChannelStatusToBeExcluded_.data(),
                                  this->ChannelStatusToBeExcluded_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.expanded_v_DB_reco_flags,
                                  this->expanded_v_DB_reco_flags_.data(),
                                  this->expanded_v_DB_reco_flags_.size() * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.expanded_Sizes_v_DB_reco_flags,
                                  this->expanded_Sizes_v_DB_reco_flags_.data(),
                                  this->expanded_Sizes_v_DB_reco_flags_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.expanded_flagbit_v_DB_reco_flags,
                                  this->expanded_flagbit_v_DB_reco_flags_.data(),
                                  this->expanded_flagbit_v_DB_reco_flags_.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
      });
  return product;
}

TYPELOOKUP_DATA_REG(EcalRecHitParametersGPU);
