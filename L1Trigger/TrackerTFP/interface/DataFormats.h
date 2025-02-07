#ifndef L1Trigger_TrackerTFP_DataFormats_h
#define L1Trigger_TrackerTFP_DataFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Track Trigger emulator
enabling automated conversions from frames to stubs/tracks and vice versa
In data members of classes Stub* & Track* below, the variables describing
stubs/tracks are stored both in digitial format as a 64b word in frame_,
and in undigitized format in an std::tuple. (This saves CPU)
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <vector>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <iostream>
#include <string>

namespace trackerTFP {

  // track trigger processes
  enum class Process { begin, dtc = begin, pp, gp, ht, ctb, kf, dr, tfp, end, x };
  // track trigger variables
  enum class Variable { begin, r = begin, phi, z, dPhi, dZ, inv2R, phiT, cot, zT, layer, match, end, x };
  // track trigger process order
  constexpr std::initializer_list<Process> Processes = {
      Process::dtc, Process::pp, Process::gp, Process::ht, Process::ctb, Process::kf, Process::dr, Process::tfp};
  // conversion: Process to int
  inline constexpr int operator+(Process p) { return static_cast<int>(p); }
  // conversion: Variable to int
  inline constexpr int operator+(Variable v) { return static_cast<int>(v); }
  inline constexpr Process operator+(Process p, int i) { return Process(+p + i); }
  inline constexpr Variable operator+(Variable v, int i) { return Variable(+v + i); }

  //Base class representing format of a variable
  class DataFormat {
  public:
    DataFormat() {}
    DataFormat(bool twos, bool biased = true) : twos_(twos), width_(0), base_(1.), range_(0.) {}
    DataFormat(bool twos, int width, double base, double range)
        : twos_(twos), width_(width), base_(base), range_(range) {}
    virtual ~DataFormat() {}
    // converts int to bitvector
    TTBV ttBV(int i) const { return TTBV(i, width_, twos_); }
    // converts double to bitvector
    TTBV ttBV(double d) const { return TTBV(d, base_, width_, twos_); }
    // extracts int from bitvector, removing these bits from bitvector
    void extract(TTBV& in, int& out) const { out = in.extract(width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, double& out) const { out = in.extract(base_, width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, TTBV& out) const { out = in.slice(width_, twos_); }
    // extracts bool from bitvector, removing these bits from bitvector
    void extract(TTBV& in, bool& out) const { out = in.extract(); }
    // attaches integer to bitvector
    void attach(const int i, TTBV& ttBV) const { ttBV += TTBV(i, width_, twos_); }
    // attaches double to bitvector
    void attach(const double d, TTBV& ttBV) const { ttBV += TTBV(d, base_, width_, twos_); }
    // attaches bitvector to bitvector
    void attach(const TTBV& bv, TTBV& ttBV) const { ttBV += bv; }
    // converts int to double
    double floating(int i) const { return (i + .5) * base_; }
    // converts double to int
    int integer(double d) const { return std::floor(d / base_ + 1.e-12); }
    // converts double to int and back to double
    double digi(double d) const { return floating(integer(d)); }
    // converts binary integer value to twos complement integer value
    int toSigned(int i) const { return i - std::pow(2, width_) / 2; }
    // converts twos complement integer value to binary integer value
    int toUnsigned(int i) const { return i + std::pow(2, width_) / 2; }
    // converts floating point value to binary integer value
    int toUnsigned(double d) const { return this->integer(d) + std::pow(2, width_) / 2; }
    // biggest representable floating point value
    //double limit() const { return (range_ - base_) / (twos_ ? 2. : 1.); }
    // returns false if data format would oferflow for this double value
    bool inRange(double d, bool digi = true) const {
      const double range = digi ? base_ * pow(2, width_) : range_;
      return d >= -range / 2. && d < range / 2.;
    }
    // returns false if data format would oferflow for this int value
    bool inRange(int i) const { return inRange(floating(i)); }
    // true if twos'complement or false if binary representation is chosen
    bool twos() const { return twos_; }
    // number of used bits
    int width() const { return width_; }
    // precision
    double base() const { return base_; }
    // covered range
    double range() const { return range_; }

  protected:
    // true if twos'complement or false if binary representation is chosen
    bool twos_;
    // number of used bits
    int width_;
    // precision
    double base_;
    // covered range
    double range_;
  };

  // function template for DataFormat generation
  template <Variable v, Process p>
  DataFormat makeDataFormat(const tt::Setup* setup);

  // specializations

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::tfp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::tfp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tfp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::tfp>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::r, Process::dtc>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::dtc>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z, Process::dtc>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::dtc>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::phi, Process::gp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z, Process::gp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::gp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::gp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::gp>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::gp>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::phi, Process::ht>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::ht>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::ht>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::dPhi, Process::ctb>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::dZ, Process::ctb>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::ctb>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::kf>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::kf>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::kf>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::kf>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::kf>(const tt::Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::match, Process::kf>(const tt::Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::cot, Process::dr>(const tt::Setup* setup);

  /*! \class  trackerTFP::DataFormats
   *  \brief  Class to calculate and provide dataformats used by Track Trigger emulator
   *  \author Thomas Schuh
   *  \date   2020, June
   */
  class DataFormats {
  private:
    // variable flavour mapping, Each row below declares which processing steps use the variable named in the comment at the end of the row
    static constexpr std::array<std::array<Process, +Process::end>, +Variable::end> config_ = {{
        //  Process::dtc  Process::pp   Process::gp   Process::ht   Process::ctb  Process::kf   Process::dr,  Process::tfp
        {{Process::dtc,
          Process::dtc,
          Process::dtc,
          Process::dtc,
          Process::dtc,
          Process::dtc,
          Process::dtc,
          Process::x}},  // Variable::r
        {{Process::dtc,
          Process::dtc,
          Process::gp,
          Process::ht,
          Process::ht,
          Process::kf,
          Process::kf,
          Process::x}},  // Variable::phi
        {{Process::dtc,
          Process::dtc,
          Process::gp,
          Process::gp,
          Process::gp,
          Process::gp,
          Process::gp,
          Process::x}},  // Variable::z
        {{Process::x,
          Process::x,
          Process::x,
          Process::x,
          Process::ctb,
          Process::ctb,
          Process::ctb,
          Process::x}},  // Variable::dPhi
        {{Process::x,
          Process::x,
          Process::x,
          Process::x,
          Process::ctb,
          Process::ctb,
          Process::ctb,
          Process::x}},  // Variable::dZ
        {{Process::ht,
          Process::ht,
          Process::ht,
          Process::ht,
          Process::ht,
          Process::kf,
          Process::kf,
          Process::tfp}},  // Variable::inv2R
        {{Process::gp,
          Process::gp,
          Process::gp,
          Process::ht,
          Process::ht,
          Process::kf,
          Process::kf,
          Process::tfp}},  // Variable::phiT
        {{Process::x,
          Process::x,
          Process::gp,
          Process::x,
          Process::gp,
          Process::kf,
          Process::dr,
          Process::tfp}},  // Variable::cot
        {{Process::gp,
          Process::gp,
          Process::gp,
          Process::gp,
          Process::gp,
          Process::kf,
          Process::kf,
          Process::tfp}},  // Variable::zT
        {{Process::dtc,
          Process::dtc,
          Process::gp,
          Process::gp,
          Process::ctb,
          Process::x,
          Process::x,
          Process::x}},  // Variable::layer
        {{Process::x, Process::x, Process::x, Process::x, Process::x, Process::kf, Process::x, Process::x}}  // Variable::match
    }};
    // stub word assembly, shows which stub variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> stubs_ = {{
        {Variable::r,
         Variable::phi,
         Variable::z,
         Variable::layer,
         Variable::phiT,
         Variable::phiT,
         Variable::zT,
         Variable::zT,
         Variable::inv2R,
         Variable::inv2R},  // Process::dtc
        {Variable::r,
         Variable::phi,
         Variable::z,
         Variable::layer,
         Variable::phiT,
         Variable::phiT,
         Variable::zT,
         Variable::zT,
         Variable::inv2R,
         Variable::inv2R},                                                                             // Process::pp
        {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::inv2R, Variable::inv2R},  // Process::gp
        {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::phiT, Variable::zT},      // Process::ht
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},                       // Process::ctb
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},                       // Process::kf
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},                       // Process::dr
        {}                                                                                             // Process::tfp
    }};
    // track word assembly, shows which track variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> tracks_ = {{
        {},                                                                               // Process::dtc
        {},                                                                               // Process::pp
        {},                                                                               // Process::gp
        {},                                                                               // Process::ht
        {Variable::inv2R, Variable::phiT, Variable::zT},                                  // Process::ctb
        {Variable::inv2R, Variable::phiT, Variable::cot, Variable::zT, Variable::match},  // Process::kf
        {Variable::inv2R, Variable::phiT, Variable::cot, Variable::zT},                   // Process::dr
        {}                                                                                // Process::tfp
    }};

  public:
    DataFormats();
    DataFormats(const tt::Setup* setup);
    ~DataFormats() = default;
    // converts bits to ntuple of variables
    template <typename... Ts>
    void convertStub(Process p, const tt::Frame& bv, std::tuple<Ts...>& data) const {
      TTBV ttBV(bv);
      extractStub(p, ttBV, data);
    }
    // converts ntuple of variables to bits
    template <typename... Ts>
    void convertStub(Process p, const std::tuple<Ts...>& data, tt::Frame& bv) const {
      TTBV ttBV(1, numUnusedBitsStubs_[+p]);
      attachStub(p, data, ttBV);
      bv = ttBV.bs();
    }
    // converts bits to ntuple of variables
    template <typename... Ts>
    void convertTrack(Process p, const tt::Frame& bv, std::tuple<Ts...>& data) const {
      TTBV ttBV(bv);
      extractTrack(p, ttBV, data);
    }
    // converts ntuple of variables to bits
    template <typename... Ts>
    void convertTrack(Process p, const std::tuple<Ts...>& data, tt::Frame& bv) const {
      TTBV ttBV(1, numUnusedBitsTracks_[+p]);
      attachTrack(p, data, ttBV);
      bv = ttBV.bs();
    }
    // access to run-time constants
    const tt::Setup* setup() const { return setup_; }
    // number of bits being used for specific variable flavour
    int width(Variable v, Process p) const { return formats_[+v][+p]->width(); }
    // precision being used for specific variable flavour
    double base(Variable v, Process p) const { return formats_[+v][+p]->base(); }
    // covered range for specific variable flavour
    double range(Variable v, Process p) const { return formats_[+v][+p]->range(); }
    // number of unused frame bits for a given Stub flavour
    int numUnusedBitsStubs(Process p) const { return numUnusedBitsStubs_[+p]; }
    // number of unused frame bits for a given Track flavour
    int numUnusedBitsTracks(Process p) const { return numUnusedBitsTracks_[+p]; }
    // number of channels of a given process on a TFP
    int numChannel(Process p) const { return numChannel_[+p]; }
    // number of stub channels of a given process for whole system
    int numStreamsStubs(Process p) const { return numStreamsStubs_[+p]; }
    // number of track channels of a given process for whole system
    int numStreamsTracks(Process p) const { return numStreamsTracks_[+p]; }
    // access to spedific format
    const DataFormat& format(Variable v, Process p) const { return *formats_[+v][+p]; }

  private:
    // number of unique data formats
    int numDataFormats_;
    // method to count number of unique data formats
    template <Variable v = Variable::begin, Process p = Process::begin>
    void countFormats();
    // constructs data formats of all unique used variables and flavours
    template <Variable v = Variable::begin, Process p = Process::begin>
    void fillDataFormats();
    // helper (loop) data formats of all unique used variables and flavours
    template <Variable v, Process p, Process it = Process::begin>
    void fillFormats();
    // helper (loop) to convert bits to ntuple of variables
    template <int it = 0, typename... Ts>
    void extractStub(Process p, TTBV& ttBV, std::tuple<Ts...>& data) const {
      Variable v = *std::next(stubs_[+p].begin(), sizeof...(Ts) - 1 - it);
      formats_[+v][+p]->extract(ttBV, std::get<sizeof...(Ts) - 1 - it>(data));
      if constexpr (it + 1 != sizeof...(Ts))
        extractStub<it + 1>(p, ttBV, data);
    }
    // helper (loop) to convert bits to ntuple of variables
    template <int it = 0, typename... Ts>
    void extractTrack(Process p, TTBV& ttBV, std::tuple<Ts...>& data) const {
      Variable v = *std::next(tracks_[+p].begin(), sizeof...(Ts) - 1 - it);
      formats_[+v][+p]->extract(ttBV, std::get<sizeof...(Ts) - 1 - it>(data));
      if constexpr (it + 1 != sizeof...(Ts))
        extractTrack<it + 1>(p, ttBV, data);
    }
    // helper (loop) to convert ntuple of variables to bits
    template <int it = 0, typename... Ts>
    void attachStub(Process p, const std::tuple<Ts...>& data, TTBV& ttBV) const {
      Variable v = *std::next(stubs_[+p].begin(), it);
      formats_[+v][+p]->attach(std::get<it>(data), ttBV);
      if constexpr (it + 1 != sizeof...(Ts))
        attachStub<it + 1>(p, data, ttBV);
    }
    // helper (loop) to convert ntuple of variables to bits
    template <int it = 0, typename... Ts>
    void attachTrack(Process p, const std::tuple<Ts...>& data, TTBV& ttBV) const {
      Variable v = *std::next(tracks_[+p].begin(), it);
      formats_[+v][+p]->attach(std::get<it>(data), ttBV);
      if constexpr (it + 1 != sizeof...(Ts))
        attachTrack<it + 1>(p, data, ttBV);
    }
    // stored run-time constants
    const tt::Setup* setup_;
    // collection of unique formats
    std::vector<DataFormat> dataFormats_;
    // variable flavour mapping
    std::vector<std::vector<DataFormat*>> formats_;
    // number of unused frame bits for a all Stub flavours
    std::vector<int> numUnusedBitsStubs_;
    // number of unused frame bits for a all Track flavours
    std::vector<int> numUnusedBitsTracks_;
    // number of channels of all processes on a TFP
    std::vector<int> numChannel_;
    // number of stub channels of all processes for whole system
    std::vector<int> numStreamsStubs_;
    // number of track channels of all processes for whole system
    std::vector<int> numStreamsTracks_;
  };

  // base class to represent stubs
  template <typename... Ts>
  class Stub {
  public:
    // construct Stub from Frame
    Stub(const tt::FrameStub& fs, const DataFormats* df, Process p) : dataFormats_(df), p_(p), frame_(fs) {
      dataFormats_->convertStub(p_, frame_.second, data_);
    }
    template <typename... Others>
    // construct Stub from other Stub
    Stub(const Stub<Others...>& stub, Ts... data)
        : dataFormats_(stub.dataFormats()), p_(stub.p() + 1), frame_(stub.frame()), data_(data...) {
      dataFormats_->convertStub(p_, data_, frame_.second);
    }
    // construct Stub from TTStubRef
    Stub(const TTStubRef& ttStubRef, const DataFormats* df, Process p, Ts... data)
        : dataFormats_(df), p_(p), frame_(ttStubRef, tt::Frame()), data_(data...) {
      dataFormats_->convertStub(p_, data_, frame_.second);
    }
    Stub() {}
    ~Stub() {}
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // stub flavour
    Process p() const { return p_; }
    // acess to frame
    const tt::FrameStub& frame() const { return frame_; }

  protected:
    // all dataformats
    const DataFormats* dataFormats_;
    // stub flavour
    Process p_;
    // underlying TTStubRef and bitvector
    tt::FrameStub frame_;
    // ntuple of variables this stub is assemled of
    std::tuple<Ts...> data_;
  };

  // class to represent stubs generated by process DTC
  class StubDTC : public Stub<double, double, double, TTBV, int, int, int, int, int, int> {
  public:
    // construct StubDTC from TTStubRef
    StubDTC(const TTStubRef& ttStubRef,
            const DataFormats* df,
            double r,
            double phi,
            double z,
            const TTBV& layer,
            int phiTMin,
            int phiTMax,
            int zTMin,
            int zTMax,
            int inv2RMin,
            int inv2RMax)
        : Stub(ttStubRef, df, Process::dtc, r, phi, z, layer, phiTMin, phiTMax, zTMin, zTMax, inv2RMin, inv2RMax) {}
    ~StubDTC() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt processing nonant centre
    double phi() const { return std::get<1>(data_); }
    // stub z
    double z() const { return std::get<2>(data_); }
    // enhanced layer id
    TTBV layer() const { return std::get<3>(data_); }
    // first phi sector this stub belongs to
    int phiTMin() const { return std::get<4>(data_); }
    // last phi sector this stub belongs to
    int phiTMax() const { return std::get<5>(data_); }
    // first eta sector this stub belongs to
    int zTMin() const { return std::get<6>(data_); }
    // last eta sector this stub belongs to
    int zTMax() const { return std::get<7>(data_); }
    // first inv2R bin this stub belongs to
    int inv2RMin() const { return std::get<8>(data_); }
    // last inv2R bin this stub belongs to
    int inv2RMax() const { return std::get<9>(data_); }
  };

  // class to represent stubs generated by process patch pannel
  class StubPP : public Stub<double, double, double, TTBV, int, int, int, int, int, int> {
  public:
    // construct StubPP from Frame
    StubPP(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::pp) {}
    ~StubPP() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt processing nonant centre
    double phi() const { return std::get<1>(data_); }
    // stub z
    double z() const { return std::get<2>(data_); }
    // enhanced layer id
    const TTBV& layer() const { return std::get<3>(data_); }
    // first phi sector this stub belongs to
    int phiTMin() const { return std::get<4>(data_); }
    // last phi sector this stub belongs to
    int phiTMax() const { return std::get<5>(data_); }
    // first eta sector this stub belongs to
    int zTMin() const { return std::get<6>(data_); }
    // last eta sector this stub belongs to
    int zTMax() const { return std::get<7>(data_); }
    // first inv2R bin this stub belongs to
    int inv2RMin() const { return std::get<8>(data_); }
    // last inv2R bin this stub belongs to
    int inv2RMax() const { return std::get<9>(data_); }
  };

  // class to represent stubs generated by process geometric processor
  class StubGP : public Stub<double, double, double, TTBV, int, int> {
  public:
    // construct StubGP from Frame
    StubGP(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::gp) {}
    // construct StubGP from StubPP
    StubGP(const StubPP& stub, double r, double phi, double z, const TTBV& layer, int inv2RMin, int inv2RMax)
        : Stub(stub, r, phi, z, layer, inv2RMin, inv2RMax) {}
    ~StubGP() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt phi sector centre
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); }
    // enhanced layer id
    const TTBV& layer() const { return std::get<3>(data_); }
    // first inv2R bin this stub belongs to
    int inv2RMin() const { return std::get<4>(data_); }
    // last inv2R bin this stub belongs to
    int inv2RMax() const { return std::get<5>(data_); }
  };

  // class to represent stubs generated by process hough transform
  class StubHT : public Stub<double, double, double, TTBV, int, int> {
  public:
    // construct StubHT from Frame
    StubHT(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::ht) {}
    // construct StubHT from StubGP
    StubHT(const StubGP& stub, double r, double phi, double z, const TTBV& layer, int phiT, int zT)
        : Stub(stub, r, phi, z, layer, phiT, zT) {}
    ~StubHT() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); };
    // stub phi residual wrt track parameter
    double phi() const { return std::get<1>(data_); };
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); };
    // enhanced layer id
    const TTBV& layer() const { return std::get<3>(data_); }
    // stub phi at radius chosenRofPhi wrt processing nonant centre
    int phiT() const { return std::get<4>(data_); };
    // eta sector
    int zT() const { return std::get<5>(data_); };
  };

  // class to represent stubs generated by process CTB
  class StubCTB : public Stub<double, double, double, double, double> {
  public:
    // construct StubTB from Frame
    StubCTB(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::ctb) {}
    // construct StubTB from StubZHT
    StubCTB(const StubHT& stub, double r, double phi, double z, double dPhi, double dZ)
        : Stub(stub, r, phi, z, dPhi, dZ) {}
    ~StubCTB() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt finer track parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt track parameter
    double z() const { return std::get<2>(data_); }
    // stub phi uncertainty
    double dPhi() const { return std::get<3>(data_); }
    // stub z uncertainty
    double dZ() const { return std::get<4>(data_); }
  };

  // class to represent stubs generated by process kalman filter
  class StubKF : public Stub<double, double, double, double, double> {
  public:
    // construct StubKF from Frame
    StubKF(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::kf) {}
    // construct StubKF from StubCTB
    StubKF(const StubCTB& stub, double r, double phi, double z, double dPhi, double dZ)
        : Stub(stub, r, phi, z, dPhi, dZ) {}
    ~StubKF() {}
    // stub radius wrt choenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt fitted parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt fitted parameter
    double z() const { return std::get<2>(data_); }
    // stub phi uncertainty
    double dPhi() const { return std::get<3>(data_); }
    // stub z uncertainty
    double dZ() const { return std::get<4>(data_); }
  };

  // class to represent stubs generated by process duplicate removal
  class StubDR : public Stub<double, double, double, double, double> {
  public:
    // construct StubDR from Frame
    StubDR(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::dr) {}
    // construct StubDR from StubKF
    StubDR(const StubKF& stub, double r, double phi, double z, double dPhi, double dZ)
        : Stub(stub, r, phi, z, dPhi, dZ) {}
    ~StubDR() {}
    // stub radius wrt choenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt fitted parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt fitted parameter
    double z() const { return std::get<2>(data_); }
    // stub phi uncertainty
    double dPhi() const { return std::get<3>(data_); }
    // stub z uncertainty
    double dZ() const { return std::get<4>(data_); }
  };

  // base class to represent tracks
  template <typename... Ts>
  class Track {
  public:
    // construct Track from Frame
    Track(const tt::FrameTrack& ft, const DataFormats* df, Process p) : dataFormats_(df), p_(p), frame_(ft) {
      dataFormats_->convertTrack(p_, frame_.second, data_);
    }
    // construct Track from TTTrackRef
    Track(const TTTrackRef& ttTrackRef, const DataFormats* df, Process p, Ts... data)
        : dataFormats_(df), p_(p), frame_(ttTrackRef, tt::Frame()), data_(data...) {
      dataFormats_->convertTrack(p_, data_, frame_.second);
    }
    // construct Track from other Track
    template <typename... Others>
    Track(const Track<Others...>& track, Ts... data)
        : dataFormats_(track.dataFormats()), p_(track.p() + 1), frame_(track.frame()), data_(data...) {
      dataFormats_->convertTrack(p_, data_, frame_.second);
    }
    ~Track() {}
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // track flavour
    Process p() const { return p_; }
    // acces to frame
    const tt::FrameTrack& frame() const { return frame_; }

  protected:
    // all data formats
    const DataFormats* dataFormats_;
    // track flavour
    Process p_;
    // underlying TTTrackRef and bitvector
    tt::FrameTrack frame_;
    // ntuple of variables this track is assemled of
    std::tuple<Ts...> data_;
  };

  // class to represent tracks generated by process Clean Track Builder
  class TrackCTB : public Track<double, double, double> {
  public:
    // construct TrackTB from Frame
    TrackCTB(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::ctb) {}
    // construct TrackTB from StubsCTB
    TrackCTB(const TTTrackRef& tTTrackRef, const DataFormats* df, double inv2R, double phiT, double zT)
        : Track(tTTrackRef, df, Process::ctb, inv2R, phiT, zT) {}
    ~TrackCTB() {}
    // track inv2R
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius chosenRofPhi wrt pprocessing centre
    double phiT() const { return std::get<1>(data_); }
    // track z at radius chosenRofZ
    double zT() const { return std::get<2>(data_); }
  };

  // class to represent tracks generated by process kalman filter
  class TrackKF : public Track<double, double, double, double, TTBV> {
  public:
    // construct TrackKF from Frame
    TrackKF(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::kf) {}
    // construct TrackKF from TrackCTB
    TrackKF(const TrackCTB& track, double inv2R, double phiT, double cot, double zT, const TTBV& match)
        : Track(track, inv2R, phiT, cot, zT, match) {}
    ~TrackKF() {}
    // track qOver pt
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius chosenRofPhi wrt processing nonant centre
    double phiT() const { return std::get<1>(data_); }
    // track cotTheta wrt eta sector cotTheta
    double cot() const { return std::get<2>(data_); }
    // track z at radius chosenRofZ
    double zT() const { return std::get<3>(data_); }
    // true if kf prameter consistent with mht parameter
    const TTBV& match() const { return std::get<4>(data_); }
  };

  // class to represent tracks generated by process duplicate removal
  class TrackDR : public Track<double, double, double, double> {
  public:
    // construct TrackDR from Frame
    TrackDR(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::dr) {}
    // construct TrackDR from TrackKF
    TrackDR(const TrackKF& track, double inv2R, double phiT, double cot, double zT)
        : Track(track, inv2R, phiT, cot, zT) {}
    ~TrackDR() {}
    // track inv2R
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius 0 wrt processing nonant centre
    double phiT() const { return std::get<1>(data_); }
    // track cotThea
    double cot() const { return std::get<2>(data_); }
    // track z at radius 0
    double zT() const { return std::get<3>(data_); }
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::DataFormats, trackerTFP::DataFormatsRcd);

#endif
