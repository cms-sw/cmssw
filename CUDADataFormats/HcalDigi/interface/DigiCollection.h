#ifndef CUDADataFormats_HcalDigi_interface_DigiCollection_h
#define CUDADataFormats_HcalDigi_interface_DigiCollection_h

#include "CUDADataFormats/CaloCommon/interface/Common.h"

namespace hcal {

  // FLAVOR_HE_QIE11 = 1; Phase1 upgrade
  struct Flavor1 {
    static constexpr int WORDS_PER_SAMPLE = 1;
    static constexpr int SAMPLES_PER_WORD = 1;
    static constexpr int HEADER_WORDS = 1;

    static constexpr uint8_t adc(uint16_t const* const sample_start) { return (*sample_start & 0xff); }
    static constexpr uint8_t tdc(uint16_t const* const sample_start) { return (*sample_start >> 8) & 0x3f; }
    static constexpr uint8_t soibit(uint16_t const* const sample_start) { return (*sample_start >> 14) & 0x1; }
  };

  // FLAVOR_HB_QIE11 = 3; Phase1 upgrade
  struct Flavor3 {
    static constexpr int WORDS_PER_SAMPLE = 1;
    static constexpr int SAMPLES_PER_WORD = 1;
    static constexpr int HEADER_WORDS = 1;

    static constexpr uint8_t adc(uint16_t const* const sample_start) { return (*sample_start & 0xff); }
    static constexpr uint8_t tdc(uint16_t const* const sample_start) { return ((*sample_start >> 8) & 0x3); }
    static constexpr uint8_t soibit(uint16_t const* const sample_start) { return ((*sample_start >> 14) & 0x1); }
    static constexpr uint8_t capid(uint16_t const* const sample_start) { return ((*sample_start >> 10) & 0x3); }
  };

  // FLAVOR_HB_QIE10 = 5; Phase0
  struct Flavor5 {
    static constexpr float WORDS_PER_SAMPLE = 0.5;
    static constexpr int SAMPLES_PER_WORD = 2;
    static constexpr int HEADER_WORDS = 1;

    static constexpr uint8_t adc(uint16_t const* const sample_start, uint8_t const shifter) {
      return ((*sample_start >> shifter * 8) & 0x7f);
    }
  };

  template <typename Flavor>
  constexpr uint8_t capid_for_sample(uint16_t const* const dfstart, uint32_t const sample) {
    auto const capid_first = (*dfstart >> 8) & 0x3;
    return (capid_first + sample) & 0x3;  // same as % 4
  }

  template <>
  constexpr uint8_t capid_for_sample<Flavor3>(uint16_t const* const dfstart, uint32_t const sample) {
    return Flavor3::capid(dfstart + Flavor3::HEADER_WORDS + sample * Flavor3::WORDS_PER_SAMPLE);
  }

  template <typename Flavor>
  constexpr uint8_t soibit_for_sample(uint16_t const* const dfstart, uint32_t const sample) {
    return Flavor::soibit(dfstart + Flavor::HEADER_WORDS + sample * Flavor::WORDS_PER_SAMPLE);
  }

  template <typename Flavor>
  constexpr uint8_t adc_for_sample(uint16_t const* const dfstart, uint32_t const sample) {
    return Flavor::adc(dfstart + Flavor::HEADER_WORDS + sample * Flavor::WORDS_PER_SAMPLE);
  }

  template <typename Flavor>
  constexpr uint8_t tdc_for_sample(uint16_t const* const dfstart, uint32_t const sample) {
    return Flavor::tdc(dfstart + Flavor::HEADER_WORDS + sample * Flavor::WORDS_PER_SAMPLE);
  }

  template <>
  constexpr uint8_t adc_for_sample<Flavor5>(uint16_t const* const dfstart, uint32_t const sample) {
    // avoid using WORDS_PER_SAMPLE and simply shift
    return Flavor5::adc(dfstart + Flavor5::HEADER_WORDS + (sample >> 1), sample % 2);
  }

  template <typename Flavor>
  constexpr uint32_t compute_stride(uint32_t const nsamples) {
    return static_cast<uint32_t>(nsamples * Flavor::WORDS_PER_SAMPLE) + Flavor::HEADER_WORDS;
  }

  template <typename Flavor>
  constexpr uint32_t compute_nsamples(uint32_t const nwords) {
    if constexpr (Flavor::SAMPLES_PER_WORD >= 1)
      return (nwords - Flavor::HEADER_WORDS) * Flavor::SAMPLES_PER_WORD;
    else
      return (nwords - Flavor::HEADER_WORDS) / Flavor::WORDS_PER_SAMPLE;
  }

  //
  template <typename StoragePolicy>
  struct DigiCollectionBase : public ::calo::common::AddSize<typename StoragePolicy::TagType> {
    DigiCollectionBase() = default;
    DigiCollectionBase(DigiCollectionBase const&) = default;
    DigiCollectionBase& operator=(DigiCollectionBase const&) = default;

    DigiCollectionBase(DigiCollectionBase&&) = default;
    DigiCollectionBase& operator=(DigiCollectionBase&&) = default;

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type resize(std::size_t size) {
      ids.resize(size);
      data.resize(size * stride);
    }

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type reserve(std::size_t size) {
      ids.reserve(size);
      data.reserve(size * stride);
    }

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type clear() {
      ids.clear();
      data.clear();
    }

    typename StoragePolicy::template StorageSelector<uint32_t>::type ids;
    typename StoragePolicy::template StorageSelector<uint16_t>::type data;
    uint32_t stride{0};
  };

  template <typename Flavor, typename StoragePolicy>
  struct DigiCollection : public DigiCollectionBase<StoragePolicy> {
    using DigiCollectionBase<StoragePolicy>::DigiCollectionBase;
  };

  // NOTE: base ctors will not be available
  template <typename StoragePolicy>
  struct DigiCollection<Flavor5, StoragePolicy> : public DigiCollectionBase<StoragePolicy> {
    DigiCollection() = default;

    DigiCollection(DigiCollection const&) = default;
    DigiCollection& operator=(DigiCollection const&) = default;

    DigiCollection(DigiCollection&&) = default;
    DigiCollection& operator=(DigiCollection&&) = default;

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type resize(std::size_t size) {
      DigiCollectionBase<StoragePolicy>::resize(size);
      npresamples.resize(size);
    }

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type reserve(std::size_t size) {
      DigiCollectionBase<StoragePolicy>::reserve(size);
      npresamples.reserve(size);
    }

    template <typename T = typename StoragePolicy::TagType>
    typename std::enable_if<std::is_same<T, ::calo::common::tags::Vec>::value, void>::type clear() {
      DigiCollectionBase<StoragePolicy>::clear();
      npresamples.clear();
    }

    // add npresamples member
    typename StoragePolicy::template StorageSelector<uint8_t>::type npresamples;
  };

}  // namespace hcal

#endif  // CUDADataFormats_HcalDigi_interface_DigiCollection_h
