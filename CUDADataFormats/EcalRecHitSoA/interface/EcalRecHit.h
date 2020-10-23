#ifndef CUDADataFormats_EcalRecHitSoA_interface_EcalRecHit_h
#define CUDADataFormats_EcalRecHitSoA_interface_EcalRecHit_h

#include <array>
#include <vector>

#include "CUDADataFormats/CaloCommon/interface/Common.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/RecoTypes.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

namespace ecal {

  template <typename StoragePolicy>
  struct RecHit : public ::calo::common::AddSize<typename StoragePolicy::TagType> {
    RecHit() = default;
    RecHit(const RecHit&) = default;
    RecHit& operator=(const RecHit&) = default;

    RecHit(RecHit&&) = default;
    RecHit& operator=(RecHit&&) = default;

    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type energy;
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type time;
    // should we remove the following, since already included in "extra" ?
    typename StoragePolicy::template StorageSelector<reco::StorageScalarType>::type chi2;
    typename StoragePolicy::template StorageSelector<uint32_t>::type
        extra;  // packed uint32_t for timeError, chi2, energyError
    typename StoragePolicy::template StorageSelector<uint32_t>::type
        flagBits;  // store rechit condition (see Flags enum) in a bit-wise way
    typename StoragePolicy::template StorageSelector<uint32_t>::type did;

    template <typename U = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<U, ::calo::common::tags::Vec>::value, void>::type resize(size_t size) {
      energy.resize(size);
      time.resize(size);
      chi2.resize(size);
      extra.resize(size);
      flagBits.resize(size);
      did.resize(size);
    }
  };

}  // namespace ecal

#endif  // CUDADataFormats_EcalRecHitSoA_interface_EcalRecHit_h
