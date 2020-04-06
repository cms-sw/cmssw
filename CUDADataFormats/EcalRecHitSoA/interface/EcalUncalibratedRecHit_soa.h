#ifndef CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_soa_h
#define CUDADataFormats_EcalRecHitSoA_interface_EcalUncalibratedRecHit_soa_h

#include <vector>
#include <array>

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"

namespace ecal {

  namespace Tag {

    struct soa {};
    struct ptr {};

  }  // namespace Tag

  namespace Detail {

    // empty base
    template <typename T>
    struct Base {};

    // add number of values for ptr case
    template <>
    struct Base<::ecal::Tag::ptr> {
      uint32_t size;
    };

  }  // namespace Detail

  template <typename T, typename L = Tag::soa>
  struct type_wrapper {
    using type = std::vector<T, CUDAHostAllocator<T>>;
  };

  template <typename T>
  struct type_wrapper<T, Tag::ptr> {
    using type = T*;
  };

  template <typename L = Tag::soa>
  struct UncalibratedRecHit : public Detail::Base<L> {
    UncalibratedRecHit() = default;
    UncalibratedRecHit(const UncalibratedRecHit&) = default;
    UncalibratedRecHit& operator=(const UncalibratedRecHit&) = default;

    UncalibratedRecHit(UncalibratedRecHit&&) = default;
    UncalibratedRecHit& operator=(UncalibratedRecHit&&) = default;

    // TODO: std::array causes root's dictionary problems
    typename type_wrapper<reco::ComputationScalarType, L>::type amplitudesAll;
    //    typename type_wrapper<std::array<reco::ComputationScalarType,
    //        EcalDataFrame::MAXSAMPLES>, L>::type amplitudesAll;
    typename type_wrapper<reco::StorageScalarType, L>::type amplitude;
    typename type_wrapper<reco::StorageScalarType, L>::type chi2;
    typename type_wrapper<reco::StorageScalarType, L>::type pedestal;
    typename type_wrapper<reco::StorageScalarType, L>::type jitter;
    typename type_wrapper<reco::StorageScalarType, L>::type jitterError;
    typename type_wrapper<uint32_t, L>::type did;
    typename type_wrapper<uint32_t, L>::type flags;

    template <typename U = L>
    typename std::enable_if<std::is_same<U, Tag::soa>::value, void>::type resize(size_t size) {
      amplitudesAll.resize(size * EcalDataFrame::MAXSAMPLES);
      amplitude.resize(size);
      pedestal.resize(size);
      chi2.resize(size);
      did.resize(size);
      flags.resize(size);
      jitter.resize(size);
      jitterError.resize(size);
    }
  };

  using SoAUncalibratedRecHitCollection = UncalibratedRecHit<Tag::soa>;

}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalUncalibratedRecHit_soa_h
