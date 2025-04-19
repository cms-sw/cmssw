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
  inline constexpr VariableKF operator+(VariableKF v, int i) { return VariableKF(+v + i); }

  // Configuration
  struct ConfigKF {
    bool enableIntegerEmulation_;
    int widthR00_;
    int widthR11_;
    int widthC00_;
    int widthC01_;
    int widthC11_;
    int widthC22_;
    int widthC23_;
    int widthC33_;
    int baseShiftx0_;
    int baseShiftx1_;
    int baseShiftx2_;
    int baseShiftx3_;
    int baseShiftr0_;
    int baseShiftr1_;
    int baseShiftS00_;
    int baseShiftS01_;
    int baseShiftS12_;
    int baseShiftS13_;
    int baseShiftR00_;
    int baseShiftR11_;
    int baseShiftInvR00Approx_;
    int baseShiftInvR11Approx_;
    int baseShiftInvR00Cor_;
    int baseShiftInvR11Cor_;
    int baseShiftInvR00_;
    int baseShiftInvR11_;
    int baseShiftS00Shifted_;
    int baseShiftS01Shifted_;
    int baseShiftS12Shifted_;
    int baseShiftS13Shifted_;
    int baseShiftK00_;
    int baseShiftK10_;
    int baseShiftK21_;
    int baseShiftK31_;
    int baseShiftC00_;
    int baseShiftC01_;
    int baseShiftC11_;
    int baseShiftC22_;
    int baseShiftC23_;
    int baseShiftC33_;
    int baseShiftr0Shifted_;
    int baseShiftr1Shifted_;
    int baseShiftr02_;
    int baseShiftr12_;
    int baseShiftchi20_;
    int baseShiftchi21_;
  };

  class DataFormatKF {
  public:
    DataFormatKF(const VariableKF& v, bool twos, bool enableIntegerEmulation, int width, double base, double range);
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
    KalmanFilterFormats();
    ~KalmanFilterFormats() = default;
    void beginRun(const DataFormats* dataFormats, const ConfigKF& iConfig);
    const tt::Setup* setup() const { return dataFormats_->setup(); }
    const DataFormats* dataFormats() const { return dataFormats_; }
    DataFormatKF& format(VariableKF v) { return formats_[+v]; }
    void endJob(std::stringstream& ss);

  private:
    template <VariableKF it = VariableKF::begin>
    void fillFormats();
    ConfigKF iConfig_;
    const DataFormats* dataFormats_;
    std::vector<DataFormatKF> formats_;
  };

  // function template for DataFormat generation
  template <VariableKF v>
  DataFormatKF makeDataFormat(const DataFormats* dataFormats, const ConfigKF& iConfig);

  template <>
  DataFormatKF makeDataFormat<VariableKF::x0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::x1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::x2>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::x3>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::H00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::H12>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::m0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::m1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::v0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::v1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S01>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S12>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S13>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S00Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S01Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S12Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::S13Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::K00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::K10>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::K21>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::K31>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::R00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::R11>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::R00Rough>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::R11Rough>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00Approx>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11Approx>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00Cor>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11Cor>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C00>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C01>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C11>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C22>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C23>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::C33>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r0Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r1Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r02>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::r12>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::chi20>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::chi21>(const DataFormats* dataFormats, const ConfigKF& iConfig);

  template <>
  DataFormatKF makeDataFormat<VariableKF::dH>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invdH>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::invdH2>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::H2>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::Hm0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::Hm1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::Hv0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::Hv1>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::H2v0>(const DataFormats* dataFormats, const ConfigKF& iConfig);
  template <>
  DataFormatKF makeDataFormat<VariableKF::H2v1>(const DataFormats* dataFormats, const ConfigKF& iConfig);

}  // namespace trackerTFP

#endif
