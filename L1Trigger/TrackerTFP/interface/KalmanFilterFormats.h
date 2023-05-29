#ifndef L1Trigger_TrackerTFP_KalmanFilterFormats_h
#define L1Trigger_TrackerTFP_KalmanFilterFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Kalman Filter emulator
enabling tuning of bit widths
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormatsRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <array>
#include <string>

namespace trackerTFP {

  enum class VariableKF {
    begin,
    x0 = begin,
    x1,
    x2,
    x3,
    H00,
    H12,
    m0,
    m1,
    v0,
    v1,
    r0,
    r1,
    S00,
    S01,
    S12,
    S13,
    K00,
    K10,
    K21,
    K31,
    R00,
    R11,
    R00Rough,
    R11Rough,
    invR00Approx,
    invR11Approx,
    invR00Cor,
    invR11Cor,
    invR00,
    invR11,
    C00,
    C01,
    C11,
    C22,
    C23,
    C33,
    end,
    x
  };
  inline constexpr int operator+(VariableKF v) { return static_cast<int>(v); }
  inline constexpr VariableKF operator++(VariableKF v) { return VariableKF(+v + 1); }

  class DataFormatKF {
  public:
    DataFormatKF(const VariableKF& v, bool twos);
    virtual ~DataFormatKF() {}
    double digi(double val) const { return (std::floor(val / base_ + 1.e-12) + .5) * base_; }
    bool twos() const { return twos_; }
    int width() const { return width_; }
    double base() const { return base_; }
    double range() const { return range_; }
    const std::pair<double, double>& rangeActual() const { return rangeActual_; }
    // returns false if data format would oferflow for this double value
    bool inRange(double d) const;
    void updateRangeActual(double d);
    int integer(double d) const { return floor(d / base_); }

  protected:
    VariableKF v_;
    bool twos_;
    int width_;
    double base_;
    double range_;
    std::pair<double, double> rangeActual_;
  };

  template <VariableKF v>
  class FormatKF : public DataFormatKF {
  public:
    FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
    ~FormatKF() override {}

  private:
    void calcRange() { range_ = base_ * pow(2, width_); }
  };

  template <>
  FormatKF<VariableKF::x0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::x1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::x2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::x3>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::H00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::H12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::m0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::m1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::v0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::v1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::r0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::r1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S13>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::K00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::K10>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::K21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::K31>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::R00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::R11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::R00Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::R11Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR00Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR11Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR00Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR11Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invR11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C22>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C23>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::C33>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);

  class KalmanFilterFormats {
  public:
    KalmanFilterFormats();
    KalmanFilterFormats(const edm::ParameterSet& iConfig, const DataFormats* dataFormats);
    ~KalmanFilterFormats() {}
    const tt::Setup* setup() const { return setup_; }
    const DataFormats* dataFormats() const { return dataFormats_; }
    int width(VariableKF v) const { return formats_[+v].width(); }
    double base(VariableKF v) const { return formats_[+v].base(); }
    DataFormatKF& format(VariableKF v) { return formats_[+v]; }
    void endJob();

  private:
    template <VariableKF it = VariableKF::begin>
    void fillFormats();
    const edm::ParameterSet iConfig_;
    const DataFormats* dataFormats_;
    const tt::Setup* setup_;
    std::vector<DataFormatKF> formats_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::KalmanFilterFormats, trackerTFP::KalmanFilterFormatsRcd);

#endif
