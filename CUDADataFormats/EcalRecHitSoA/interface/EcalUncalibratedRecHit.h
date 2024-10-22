#ifndef CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h
#define CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h

#include "CUDADataFormats/CaloCommon/interface/Common.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalRecHit/interface/RecoTypes.h"

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
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type amplitudeError;
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
      amplitudeError.resize(size);
      pedestal.resize(size);
      chi2.resize(size);
      did.resize(size);
      flags.resize(size);
      jitter.resize(size);
      jitterError.resize(size);
    }
  };

}  // namespace ecal

#endif  // CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_h
