#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <limits>
#include <cstring>

namespace trackerTFP {

  constexpr auto variableKFstrs_ = {
      "x0",         "x1",         "x2",         "x3",         "H00",          "H12",          "m0",        "m1",
      "v0",         "v1",         "r0",         "r1",         "S00",          "S01",          "S12",       "S13",
      "S00Shifted", "S01Shifted", "S12Shifted", "S13Shifted", "K00",          "K10",          "K21",       "K31",
      "R00",        "R11",        "R00Rough",   "R11Rough",   "invR00Approx", "invR11Approx", "invR00Cor", "invR11Cor",
      "invR00",     "invR11",     "C00",        "C01",        "C11",          "C22",          "C23",       "C33",
      "r0Shifted",  "r1Shifted",  "r02",        "r12",        "chi20",        "chi21"};

  void KalmanFilterFormats::endJob(std::stringstream& ss) {
    const int wName =
        std::strlen(*std::max_element(variableKFstrs_.begin(), variableKFstrs_.end(), [](const auto& a, const auto& b) {
          return std::strlen(a) < std::strlen(b);
        }));
    for (VariableKF v = VariableKF::begin; v != VariableKF::dH; v = VariableKF(+v + 1)) {
      const double r =
          format(v).twos() ? std::max(std::abs(format(v).min()), std::abs(format(v).max())) * 2. : format(v).max();
      const int delta = format(v).width() - ceil(log2(r / format(v).base()));
      ss << std::setw(wName) << *std::next(variableKFstrs_.begin(), +v) << ": ";
      ss << std::setw(3) << (delta == -2147483648 ? "-" : std::to_string(delta)) << std::endl;
    }
  }

  KalmanFilterFormats::KalmanFilterFormats() { formats_.reserve(+VariableKF::end); }

  void KalmanFilterFormats::beginRun(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    dataFormats_ = dataFormats;
    iConfig_ = iConfig;
    fillFormats();
  }

  template <VariableKF it>
  void KalmanFilterFormats::fillFormats() {
    formats_.emplace_back(makeDataFormat<it>(dataFormats_, iConfig_));
    if constexpr (it + 1 != VariableKF::end)
      fillFormats<it + 1>();
  }

  DataFormatKF::DataFormatKF(
      const VariableKF& v, bool twos, bool enableIntegerEmulation, int width, double base, double range)
      : v_(v),
        twos_(twos),
        enableIntegerEmulation_(enableIntegerEmulation),
        width_(width),
        base_(base),
        range_(range),
        min_(std::numeric_limits<double>::max()),
        abs_(std::numeric_limits<double>::max()),
        max_(std::numeric_limits<double>::lowest()) {}

  // returns false if data format would oferflow for this double value
  bool DataFormatKF::inRange(double d) const {
    if (twos_)
      return d >= -range_ / 2. && d < range_ / 2.;
    return d >= 0 && d < range_;
  }

  void DataFormatKF::updateRangeActual(double d) {
    min_ = std::min(min_, d);
    abs_ = std::min(abs_, std::abs(d));
    max_ = std::max(max_, d);
    if (enableIntegerEmulation_ && !inRange(d)) {
      std::string v = *std::next(variableKFstrs_.begin(), +v_);
      cms::Exception exception("out_of_range");
      exception.addContext("trackerTFP:DataFormatKF::updateRangeActual");
      exception << "Variable " << v << " = " << d << " is out of range " << (twos_ ? -range_ / 2. : 0) << " to "
                << (twos_ ? range_ / 2. : range_) << "." << std::endl;
      if (twos_ || d >= 0.)
        exception.addAdditionalInfo("Consider raising BaseShift" + v + " in KalmanFilterFormats_cfi.py");
      exception.addAdditionalInfo("Consider disabling integer emulation in KalmanFilterFormats_cfi.py");
      throw exception;
    }
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::x0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& input = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.baseShiftx0_;
    const double base = std::pow(2, baseShift) * input.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::x0, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::x1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& input = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftx1_;
    const double base = std::pow(2, baseShift) * input.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::x1, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::x2>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& input = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.baseShiftx2_;
    const double base = std::pow(2, baseShift) * input.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::x2, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::x3>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& input = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftx3_;
    const double base = std::pow(2, baseShift) * input.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::x3, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::H00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    const double base = ctb.base();
    const int width = ctb.width();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::H00, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::H12>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    const double base = ctb.base();
    const double rangeMin = 2. * dataFormats->setup()->maxRz();
    const int width = std::ceil(std::log2(rangeMin / base));
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::H12, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::m0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::phi, Process::ctb);
    const double base = ctb.base();
    const int width = ctb.width();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::m0, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::m1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::z, Process::ctb);
    const double base = ctb.base();
    const int width = ctb.width();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::m1, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::v0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& dPhi = dataFormats->format(Variable::dPhi, Process::ctb);
    const DataFormatKF S01 = makeDataFormat<VariableKF::S01>(dataFormats, iConfig);
    const double range = dPhi.range() * dPhi.range() * 4.;
    const double base = S01.base();
    const int width = std::ceil(std::log2(range / base) - 1.e-11);
    return DataFormatKF(VariableKF::v0, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::v1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& dZ = dataFormats->format(Variable::dZ, Process::ctb);
    const DataFormatKF S13 = makeDataFormat<VariableKF::S13>(dataFormats, iConfig);
    const double range = dZ.range() * dZ.range() * 4.;
    const double base = S13.base();
    const int width = std::ceil(std::log2(range / base) - 1.e-11);
    return DataFormatKF(VariableKF::v1, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftr0_;
    const double base = std::pow(2., baseShift) * x1.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r0, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftr1_;
    const double base = std::pow(2., baseShift) * x3.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r1, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftS00_;
    const double base = std::pow(2., baseShift) * x0.base() * x1.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S00, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S01>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftS01_;
    const double base = std::pow(2., baseShift) * x1.base() * x1.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S01, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S12>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftS12_;
    const double base = std::pow(2., baseShift) * x2.base() * x3.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S12, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S13>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftS13_;
    const double base = std::pow(2., baseShift) * x3.base() * x3.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S13, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S00Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftS00Shifted_;
    const double base = std::pow(2., baseShift) * x0.base() * x1.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S00Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S01Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftS01Shifted_;
    const double base = std::pow(2., baseShift) * x1.base() * x1.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S01Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S12Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftS12Shifted_;
    const double base = std::pow(2., baseShift) * x2.base() * x3.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S12Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::S13Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftS13Shifted_;
    const double base = std::pow(2., baseShift) * x3.base() * x3.base();
    const int width = dataFormats->setup()->widthDSPab();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::S13Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::K00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftK00_;
    const double base = std::pow(2., baseShift) * x0.base() / x1.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::K00, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::K10>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftK10_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::K10, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::K21>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftK21_;
    const double base = std::pow(2., baseShift) * x2.base() / x3.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::K21, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::K31>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftK31_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::K31, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::R00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftR00_;
    const int width = iConfig.widthR00_;
    const double base = std::pow(2., baseShift) * x1.base() * x1.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::R00, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::R11>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftR11_;
    const int width = iConfig.widthR11_;
    const double base = std::pow(2., baseShift) * x3.base() * x3.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::R11, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::R00Rough>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF R00 = makeDataFormat<VariableKF::R00>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthAddrBRAM18();
    const double range = R00.range();
    const int baseShift = R00.width() - width - 1;
    const double base = std::pow(2., baseShift) * R00.base();
    return DataFormatKF(VariableKF::R00Rough, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::R11Rough>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF R11 = makeDataFormat<VariableKF::R11>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthAddrBRAM18();
    const double range = R11.range();
    const int baseShift = R11.width() - width - 1;
    const double base = std::pow(2., baseShift) * R11.base();
    return DataFormatKF(VariableKF::R11Rough, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00Approx>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftInvR00Approx_;
    const double base = std::pow(2., baseShift) / x1.base() / x1.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR00Approx, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11Approx>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftInvR11Approx_;
    const double base = std::pow(2., baseShift) / x3.base() / x3.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR11Approx, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00Cor>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftInvR00Cor_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPau();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR00Cor, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11Cor>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftInvR11Cor_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPau();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR11Cor, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftInvR00_;
    const double base = std::pow(2., baseShift) / x1.base() / x1.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR00, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invR11>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftInvR11_;
    const double base = std::pow(2., baseShift) / x3.base() / x3.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::invR11, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C00>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.baseShiftC00_;
    const int width = iConfig.widthC00_;
    const double base = std::pow(2., baseShift) * x0.base() * x0.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C00, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C01>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftC01_;
    const int width = iConfig.widthC01_;
    const double base = std::pow(2., baseShift) * x0.base() * x1.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C01, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C11>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftC11_;
    const int width = iConfig.widthC11_;
    const double base = std::pow(2., baseShift) * x1.base() * x1.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C11, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C22>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.baseShiftC22_;
    const int width = iConfig.widthC22_;
    const double base = std::pow(2., baseShift) * x2.base() * x2.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C22, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C23>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftC23_;
    const int width = iConfig.widthC23_;
    const double base = std::pow(2., baseShift) * x2.base() * x3.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C23, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::C33>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftC33_;
    const int width = iConfig.widthC33_;
    const double base = std::pow(2., baseShift) * x3.base() * x3.base();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::C33, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r0Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftr0Shifted_;
    const double base = std::pow(2., baseShift) * x1.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r0Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r1Shifted>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftr1Shifted_;
    const double base = std::pow(2., baseShift) * x3.base();
    const int width = dataFormats->setup()->widthDSPbb();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r1Shifted, true, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r02>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.baseShiftr02_;
    const double base = std::pow(2., baseShift) * x1.base() * x1.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r02, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::r12>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.baseShiftr12_;
    const double base = std::pow(2., baseShift) * x3.base() * x3.base();
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::r12, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::chi20>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftchi20_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::chi20, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::chi21>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const int baseShift = iConfig.baseShiftchi21_;
    const double base = std::pow(2., baseShift);
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = base * std::pow(2, width);
    return DataFormatKF(VariableKF::chi21, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::dH>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    const int width = dataFormats->setup()->widthAddrBRAM18();
    const double range = dataFormats->setup()->outerRadius() - dataFormats->setup()->innerRadius();
    const double base = ctb.base() * std::pow(2, std::ceil(std::log2(range / ctb.base())) - width);
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invdH>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = 1. / dataFormats->setup()->kfMinSeedDeltaR();
    const int baseShift = std::ceil(std::log2(range * std::pow(2., -width) * H00.base()));
    const double base = std::pow(2., baseShift) / H00.base();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::invdH2>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPbu();
    const double range = 1. / pow(dataFormats->setup()->kfMinSeedDeltaR(), 2);
    const double baseH2 = H00.base() * H00.base();
    const int baseShift = std::ceil(std::log2(range * std::pow(2., -width) * baseH2));
    const double base = std::pow(2., baseShift) / baseH2;
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::H2>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const int width = H00.width() + H00.width();
    const double base = H00.base() * H00.base();
    const double range = H00.range() * H00.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::Hm0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const DataFormatKF m0 = makeDataFormat<VariableKF::m0>(dataFormats, iConfig);
    const int width = H00.width() + m0.width();
    const double base = H00.base() * m0.base();
    const double range = H00.range() * m0.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::Hm1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H12 = makeDataFormat<VariableKF::H12>(dataFormats, iConfig);
    const DataFormatKF m1 = makeDataFormat<VariableKF::m1>(dataFormats, iConfig);
    const int width = H12.width() + m1.width();
    const double base = H12.base() * m1.base();
    const double range = H12.range() * m1.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::Hv0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const DataFormatKF v0 = makeDataFormat<VariableKF::v0>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPab();
    const double base = H00.base() * v0.base() * pow(2, H00.width() + v0.width() - width);
    const double range = H00.range() * v0.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::Hv1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H12 = makeDataFormat<VariableKF::H12>(dataFormats, iConfig);
    const DataFormatKF v1 = makeDataFormat<VariableKF::v1>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPab();
    const double base = H12.base() * v1.base() * pow(2, H12.width() + v1.width() - width);
    const double range = H12.range() * v1.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::H2v0>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H00 = makeDataFormat<VariableKF::H00>(dataFormats, iConfig);
    const DataFormatKF v0 = makeDataFormat<VariableKF::v0>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPau();
    const double base = H00.base() * H00.base() * v0.base() * pow(2, 2 * H00.width() + v0.width() - width);
    const double range = H00.range() * H00.range() * v0.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

  template <>
  DataFormatKF makeDataFormat<VariableKF::H2v1>(const DataFormats* dataFormats, const ConfigKF& iConfig) {
    const DataFormatKF H12 = makeDataFormat<VariableKF::H12>(dataFormats, iConfig);
    const DataFormatKF v1 = makeDataFormat<VariableKF::v1>(dataFormats, iConfig);
    const int width = dataFormats->setup()->widthDSPau();
    const double base = H12.base() * H12.base() * v1.base() * pow(2, 2 * H12.width() + v1.width() - width);
    const double range = H12.range() * H12.range() * v1.range();
    return DataFormatKF(VariableKF::end, false, iConfig.enableIntegerEmulation_, width, base, range);
  }

}  // namespace trackerTFP
