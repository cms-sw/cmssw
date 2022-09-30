#ifndef CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h
#define CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h

#include <array>
#include <vector>

#include "CUDADataFormats/CaloCommon/interface/Common.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

namespace ecal {

  template <typename StoragePolicy>
  struct UncalibratedRecHit : public ::calo::common::AddSize<typename StoragePolicy::TagType> {
    UncalibratedRecHit() = default;
    UncalibratedRecHit(const UncalibratedRecHit&) = default;
    UncalibratedRecHit& operator=(const UncalibratedRecHit&) = default;

    UncalibratedRecHit(UncalibratedRecHit&&) = default;
    UncalibratedRecHit& operator=(UncalibratedRecHit&&) = default;

    typename StoragePolicy::template StorageSelector<reco::ComputationScalarType>::type amplitudesAll;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type amplitude;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type chi2;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type pedestal;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type jitter;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type jitterError;
    typename StoragePolicy::template StorageSelector<uint32_t>::type did;
    typename StoragePolicy::template StorageSelector<uint32_t>::type flags;

    template <typename U = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<U, ::calo::common::tags::Vec>::value, void>::type resize(size_t size) {
      amplitudesAll.resize(size * EcalDataFrame::MAXSAMPLES);
      amplitude.resize(size);
      pedestal.resize(size);
      chi2.resize(size);
      did.resize(size);
      flags.resize(size);
      jitter.resize(size);
      jitterError.resize(size);
    }

    template <typename U = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<U, ::calo::common::tags::Vec>::value, void>::type resize(size_t size,
                                                                                                  cudaStream_t stream) {
      using cms::cuda::resizeContainer;
      resizeContainer(amplitudesAll, size * EcalDataFrame::MAXSAMPLES, stream);
      resizeContainer(amplitude, size, stream);
      resizeContainer(pedestal, size, stream);
      resizeContainer(chi2, size, stream);
      resizeContainer(did, size, stream);
      resizeContainer(flags, size, stream);
      resizeContainer(jitter, size, stream);
      resizeContainer(jitterError, size, stream);
    }
  };

}  // namespace ecal

#endif  // CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h
