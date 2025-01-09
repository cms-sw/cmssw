#ifndef L1Trigger_TrackerTFP_KalmanFilterFormats_h
#define L1Trigger_TrackerTFP_KalmanFilterFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Kalman Filter emulator
enabling tuning of bit widths
----------------------------------------------------------------------*/

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
    S00Shifted,
    S01Shifted,
    S12Shifted,
    S13Shifted,
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
    r0Shifted,
    r1Shifted,
    r02,
    r12,
    chi20,
    chi21,
    dH,
    invdH,
    invdH2,
    H2,
    Hm0,
    Hm1,
    Hv0,
    Hv1,
    H2v0,
    H2v1,
    end,
    x
  };
  inline constexpr int operator+(VariableKF v) { return static_cast<int>(v); }
  inline constexpr VariableKF operator++(VariableKF v) { return VariableKF(+v + 1); }

  class DataFormatKF {
  public:
    DataFormatKF(const VariableKF& v, bool twos, const edm::ParameterSet& iConfig);
    virtual ~DataFormatKF() {}
    double digi(double val) const {
      return enableIntegerEmulation_ ? (std::floor(val / base_ + 1.e-11) + .5) * base_ : val;
    }
    bool twos() const { return twos_; }
    int width() const { return width_; }
    double base() const { return base_; }
    double range() const { return range_; }
    double min() const { return min_; }
    double abs() const { return abs_; }
    double max() const { return max_; }
    // returns false if data format would oferflow for this double value
    bool inRange(double d) const;
    void updateRangeActual(double d);
    int integer(double d) const { return floor(d / base_ + 1.e-11); }

  protected:
    VariableKF v_;
    bool twos_;
    bool enableIntegerEmulation_;
    int width_;
    double base_;
    double range_;
    double min_;
    double abs_;
    double max_;
  };

  class KalmanFilterFormats {
  public:
    KalmanFilterFormats(const edm::ParameterSet& iConfig);
    ~KalmanFilterFormats() {}
    void beginRun(const DataFormats* dataFormats);
    const tt::Setup* setup() const { return dataFormats_->setup(); }
    const DataFormats* dataFormats() const { return dataFormats_; }
    DataFormatKF& format(VariableKF v) { return formats_[+v]; }
    void endJob();

  private:
    template <VariableKF it = VariableKF::begin>
    void fillFormats();
    const edm::ParameterSet iConfig_;
    const DataFormats* dataFormats_;
    std::vector<DataFormatKF> formats_;
  };

  template <VariableKF v>
  class FormatKF : public DataFormatKF {
  public:
    FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
    ~FormatKF() override {}

  private:
    void calcRange() { range_ = base_ * pow(2, width_); }
    void calcWidth() { width_ = ceil(log2(range_ / base_) - 1.e-11); }
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
  FormatKF<VariableKF::S00Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S01Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S12Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::S13Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
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
  template <>
  FormatKF<VariableKF::r0Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::r1Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::r02>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::r12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::chi20>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::chi21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);

  template <>
  FormatKF<VariableKF::dH>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invdH>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::invdH2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::H2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::Hm0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::Hm1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::Hv0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::Hv1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::H2v0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);
  template <>
  FormatKF<VariableKF::H2v1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig);

}  // namespace trackerTFP

#endif
