#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <limits>
#include <cstring>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  constexpr auto variableKFstrs_ = {
      "x0",         "x1",         "x2",         "x3",         "H00",          "H12",          "m0",        "m1",
      "v0",         "v1",         "r0",         "r1",         "S00",          "S01",          "S12",       "S13",
      "S00Shifted", "S01Shifted", "S12Shifted", "S13Shifted", "K00",          "K10",          "K21",       "K31",
      "R00",        "R11",        "R00Rough",   "R11Rough",   "invR00Approx", "invR11Approx", "invR00Cor", "invR11Cor",
      "invR00",     "invR11",     "C00",        "C01",        "C11",          "C22",          "C23",       "C33",
      "r0Shifted",  "r1Shifted",  "r02",        "r12",        "chi20",      "chi21"};

  void KalmanFilterFormats::endJob() {
    const int wName =
        strlen(*max_element(variableKFstrs_.begin(), variableKFstrs_.end(), [](const auto& a, const auto& b) {
          return strlen(a) < strlen(b);
        }));
    for (VariableKF v = VariableKF::begin; v != VariableKF::dH; v = VariableKF(+v + 1)) {
      const double r =
          format(v).twos() ? std::max(std::abs(format(v).min()), std::abs(format(v).max())) * 2. : format(v).max();
      const int delta = format(v).width() - ceil(log2(r / format(v).base()));
      cout << setw(wName) << *next(variableKFstrs_.begin(), +v) << ": ";
      cout << setw(3) << (delta == -2147483648 ? "-" : to_string(delta)) << endl;
    }
  }

  KalmanFilterFormats::KalmanFilterFormats(const ParameterSet& iConfig) : iConfig_(iConfig) {
    formats_.reserve(+VariableKF::end);
  }

  void KalmanFilterFormats::beginRun(const DataFormats* dataFormats) {
    dataFormats_ = dataFormats;
    fillFormats();
  }

  template <VariableKF it>
  void KalmanFilterFormats::fillFormats() {
    formats_.emplace_back(FormatKF<it>(dataFormats_, iConfig_));
    if constexpr (++it != VariableKF::end)
      fillFormats<++it>();
  }

  DataFormatKF::DataFormatKF(const VariableKF& v, bool twos, const ParameterSet& iConfig)
      : v_(v),
        twos_(twos),
        enableIntegerEmulation_(iConfig.getParameter<bool>("EnableIntegerEmulation")),
        width_(0),
        base_(1.),
        range_(0.),
        min_(numeric_limits<double>::max()),
        abs_(numeric_limits<double>::max()),
        max_(numeric_limits<double>::lowest()) {}

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
      string v = *next(variableKFstrs_.begin(), +v_);
      cms::Exception exception("out_of_range");
      exception.addContext("trackerTFP:DataFormatKF::updateRangeActual");
      exception << "Variable " << v << " = " << d << " is out of range " << (twos_ ? -range_ / 2. : 0) << " to "
                << (twos_ ? range_ / 2. : range_) << "." << endl;
      if (twos_ || d >= 0.)
        exception.addAdditionalInfo("Consider raising BaseShift" + v + " in KalmanFilterFormats_cfi.py.");
      exception.addAdditionalInfo("Consider disabling integer emulation in KalmanFilterFormats_cfi.py.");
      throw exception;
    }
  }

  template <>
  FormatKF<VariableKF::x0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x0, true, iConfig) {
    const DataFormat& input = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx0");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x1, true, iConfig) {
    const DataFormat& input = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx1");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x2, true, iConfig) {
    const DataFormat& input = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx2");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x3>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x3, true, iConfig) {
    const DataFormat& input = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx3");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::H00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::H00, true, iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    base_ = ctb.base();
    width_ = ctb.width();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::H12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::H12, true, iConfig) {
    const Setup* setup = dataFormats->setup();
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    base_ = ctb.base();
    range_ = 2. * setup->maxRz();
    width_ = ceil(log2(range_ / base_));
    calcRange();
  }

  template <>
  FormatKF<VariableKF::m0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::m0, true, iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::phi, Process::ctb);
    base_ = ctb.base();
    width_ = ctb.width();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::m1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::m1, true, iConfig) {
    const DataFormat& ctb = dataFormats->format(Variable::z, Process::ctb);
    base_ = ctb.base();
    width_ = ctb.width();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::v0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::v0, false, iConfig) {
    const DataFormat& dPhi = dataFormats->format(Variable::dPhi, Process::ctb);
    const FormatKF<VariableKF::S01> S01(dataFormats, iConfig);
    range_ = dPhi.range() * dPhi.range() * 4.;
    base_ = S01.base();
    calcWidth();
  }

  template <>
  FormatKF<VariableKF::v1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::v1, false, iConfig) {
    const DataFormat& dZ = dataFormats->format(Variable::dZ, Process::ctb);
    const FormatKF<VariableKF::S13> S13(dataFormats, iConfig);
    range_ = dZ.range() * dZ.range() * 4.;
    base_ = S13.base();
    calcWidth();
  }

  template <>
  FormatKF<VariableKF::r0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r0, true, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr0");
    base_ = pow(2., baseShift) * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r1, true, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr1");
    base_ = pow(2., baseShift) * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S00, true, iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS00");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S01, true, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS01");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S12, true, iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS12");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S13>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S13, true, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS13");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S00Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S00Shifted, true, iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS00Shifted");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S01Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S01Shifted, true, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS01Shifted");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S12Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S12Shifted, true, iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS12Shifted");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S13Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S13Shifted, true, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS13Shifted");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K00, true, iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK00");
    base_ = pow(2., baseShift) * x0.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K10>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K10, true, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK10");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K21, true, iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK21");
    base_ = pow(2., baseShift) * x2.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K31>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K31, true, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK31");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R00, false, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR00");
    width_ = iConfig.getParameter<int>("WidthR00");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R11, false, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR11");
    width_ = iConfig.getParameter<int>("WidthR11");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R00Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R00Rough, false, iConfig) {
    const FormatKF<VariableKF::R00> R00(dataFormats, iConfig);
    width_ = dataFormats->setup()->widthAddrBRAM18();
    range_ = R00.range();
    const int baseShift = R00.width() - width_ - 1;
    base_ = pow(2., baseShift) * R00.base();
  }

  template <>
  FormatKF<VariableKF::R11Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R11Rough, false, iConfig) {
    const FormatKF<VariableKF::R11> R11(dataFormats, iConfig);
    width_ = dataFormats->setup()->widthAddrBRAM18();
    range_ = R11.range();
    const int baseShift = R11.width() - width_ - 1;
    base_ = pow(2., baseShift) * R11.base();
  }

  template <>
  FormatKF<VariableKF::invR00Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00Approx, false, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00Approx");
    base_ = pow(2., baseShift) / x1.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11Approx, false, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11Approx");
    base_ = pow(2., baseShift) / x3.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR00Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00Cor, false, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00Cor");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPau();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11Cor, false, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11Cor");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPau();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00, false, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00");
    base_ = pow(2., baseShift) / x1.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11, false, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11");
    base_ = pow(2., baseShift) / x3.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C00, false, iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC00");
    width_ = iConfig.getParameter<int>("WidthC00");
    base_ = pow(2., baseShift) * x0.base() * x0.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C01, true, iConfig) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC01");
    width_ = iConfig.getParameter<int>("WidthC01");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C11, false, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC11");
    width_ = iConfig.getParameter<int>("WidthC11");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C22>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C22, false, iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC22");
    width_ = iConfig.getParameter<int>("WidthC22");
    base_ = pow(2., baseShift) * x2.base() * x2.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C23>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C23, true, iConfig) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC23");
    width_ = iConfig.getParameter<int>("WidthC23");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C33>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C33, false, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC33");
    width_ = iConfig.getParameter<int>("WidthC33");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r0Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r0Shifted, true, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr0Shifted");
    base_ = pow(2., baseShift) * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r1Shifted>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r1Shifted, true, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr1Shifted");
    base_ = pow(2., baseShift) * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r02>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r02, false, iConfig) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr02");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r12, false, iConfig) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr12");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::chi20>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::chi20, false, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftchi20");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::chi21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::chi21, false, iConfig) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftchi21");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::dH>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const Setup* setup = dataFormats->setup();
    const DataFormat& ctb = dataFormats->format(Variable::r, Process::ctb);
    width_ = setup->widthAddrBRAM18();
    range_ = setup->outerRadius() - setup->innerRadius();
    base_ = ctb.base() * pow(2, ceil(log2(range_ / ctb.base())) - width_);
  }

  template <>
  FormatKF<VariableKF::invdH>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    const Setup* setup = dataFormats->setup();
    width_ = setup->widthDSPbu();
    range_ = 1. / setup->kfMinSeedDeltaR();
    const int baseShift = ceil(log2(range_ * pow(2., -width_) * H00.base()));
    base_ = pow(2., baseShift) / H00.base();
  }

  template <>
  FormatKF<VariableKF::invdH2>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    const Setup* setup = dataFormats->setup();
    width_ = setup->widthDSPbu();
    range_ = 1. / pow(setup->kfMinSeedDeltaR(), 2);
    const double baseH2 = H00.base() * H00.base();
    const int baseShift = ceil(log2(range_ * pow(2., -width_) * baseH2));
    base_ = pow(2., baseShift) / baseH2;
  }

  template <>
  FormatKF<VariableKF::H2>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    base_ = H00.base() * H00.base();
  }

  template <>
  FormatKF<VariableKF::Hm0>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, true, iConfig) {
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    const FormatKF<VariableKF::m0> m0(dataFormats, iConfig);
    base_ = H00.base() * m0.base();
  }

  template <>
  FormatKF<VariableKF::Hm1>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, true, iConfig) {
    const FormatKF<VariableKF::H12> H12(dataFormats, iConfig);
    const FormatKF<VariableKF::m1> m1(dataFormats, iConfig);
    base_ = H12.base() * m1.base();
  }

  template <>
  FormatKF<VariableKF::Hv0>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const Setup* setup = dataFormats->setup();
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    const FormatKF<VariableKF::v0> v0(dataFormats, iConfig);
    width_ = setup->widthDSPab();
    base_ = H00.base() * v0.base() * pow(2, H00.width() + v0.width() - width_);
  }

  template <>
  FormatKF<VariableKF::Hv1>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const FormatKF<VariableKF::H12> H12(dataFormats, iConfig);
    const Setup* setup = dataFormats->setup();
    const FormatKF<VariableKF::v1> v1(dataFormats, iConfig);
    width_ = setup->widthDSPab();
    base_ = H12.base() * v1.base() * pow(2, H12.width() + v1.width() - width_);
  }

  template <>
  FormatKF<VariableKF::H2v0>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const Setup* setup = dataFormats->setup();
    const FormatKF<VariableKF::H00> H00(dataFormats, iConfig);
    const FormatKF<VariableKF::v0> v0(dataFormats, iConfig);
    width_ = setup->widthDSPau();
    base_ = H00.base() * H00.base() * v0.base() * pow(2, 2 * H00.width() + v0.width() - width_);
  }

  template <>
  FormatKF<VariableKF::H2v1>::FormatKF(const DataFormats* dataFormats, const ParameterSet& iConfig)
      : DataFormatKF(VariableKF::end, false, iConfig) {
    const Setup* setup = dataFormats->setup();
    const FormatKF<VariableKF::H12> H12(dataFormats, iConfig);
    const FormatKF<VariableKF::v1> v1(dataFormats, iConfig);
    width_ = setup->widthDSPau();
    base_ = H12.base() * H12.base() * v1.base() * pow(2, 2 * H12.width() + v1.width() - width_);
  }

}  // namespace trackerTFP
