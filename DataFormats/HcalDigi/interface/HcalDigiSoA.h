#ifndef DataFormats_HcalDigi_HcalDigiSoA_h
#define DataFormats_HcalDigi_HcalDigiSoA_h

//TODO: Use Eigen column for data(?)
//#include <Eigen/Core>
//#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

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

  using QIE11dataArray = edm::StdArray<uint16_t, QIE11DigiCollection::MAXSAMPLES + Flavor1::HEADER_WORDS>;
  using QIE10dataArray = edm::StdArray<uint16_t, HBHEDataFrame::MAXSAMPLES + Flavor5::HEADER_WORDS>;

  //using QIE11dataVector = Eigen::Matrix<uint16_t,  QIE11DigiCollection::MAXSAMPLES, 1>;
  //using QIE10dataVector = Eigen::Matrix<uint16_t,  HBHEDataFrame::MAXSAMPLES, 1>;

  GENERATE_SOA_LAYOUT(HcalPhase1DigiSoALayout,
                      SOA_COLUMN(uint32_t, ids),
                      //SOA_EIGEN_COLUMN(QIE11dataVector, data),
                      SOA_COLUMN(QIE11dataArray, data),
                      SOA_SCALAR(uint32_t, stride),
                      SOA_SCALAR(uint32_t, size))
  GENERATE_SOA_LAYOUT(HcalPhase0DigiSoALayout,
                      SOA_COLUMN(uint32_t, ids),
                      SOA_COLUMN(uint32_t, npresamples),
                      //SOA_EIGEN_COLUMN(QIE10dataVector, data),
                      SOA_COLUMN(QIE10dataArray, data),
                      SOA_SCALAR(uint32_t, stride),
                      SOA_SCALAR(uint32_t, size))

  using HcalPhase1DigiSoA = HcalPhase1DigiSoALayout<>;
  using HcalPhase0DigiSoA = HcalPhase0DigiSoALayout<>;

}  // namespace hcal

#endif
