#ifndef CUDADataFormats_EcalDigi_interface_DigisCollection_h
#define CUDADataFormats_EcalDigi_interface_DigisCollection_h

namespace ecal {

  //
  // this is basically a view
  // it does not own the actual memory -> does not reclaim
  //
  struct DigisCollection {
    DigisCollection() = default;
    DigisCollection(uint32_t *ids, uint16_t *data, uint32_t ndigis) : ids{ids}, data{data}, ndigis{ndigis} {}
    DigisCollection(DigisCollection const &) = default;
    DigisCollection &operator=(DigisCollection const &) = default;

    DigisCollection(DigisCollection &&) = default;
    DigisCollection &operator=(DigisCollection &&) = default;

    // stride is statically known
    uint32_t *ids = nullptr;
    uint16_t *data = nullptr;
    uint32_t ndigis;
  };

}  // namespace ecal

#endif  // CUDADataFormats_EcalDigi_interface_DigisCollection_h
