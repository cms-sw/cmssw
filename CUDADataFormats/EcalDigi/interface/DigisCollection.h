#ifndef CUDADataFormats_EcalDigi_interface_DigisCollection_h
#define CUDADataFormats_EcalDigi_interface_DigisCollection_h

#include "CUDADataFormats/CaloCommon/interface/Common.h"

namespace ecal {

  template <typename StoragePolicy>
  struct DigisCollection : public ::calo::common::AddSize<typename StoragePolicy::TagType> {
    DigisCollection() = default;
    DigisCollection(DigisCollection const &) = default;
    DigisCollection &operator=(DigisCollection const &) = default;

    DigisCollection(DigisCollection &&) = default;
    DigisCollection &operator=(DigisCollection &&) = default;

    // stride is statically known
    typename StoragePolicy::template StorageSelector<uint32_t>::type ids;
    typename StoragePolicy::template StorageSelector<uint16_t>::type data;
  };

}  // namespace ecal

#endif  // CUDADataFormats_EcalDigi_interface_DigisCollection_h
