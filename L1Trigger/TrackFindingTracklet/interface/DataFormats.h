#ifndef L1Trigger_TrackFindingTracklet_DataFormats_h
#define L1Trigger_TrackFindingTracklet_DataFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Hybrid emulator
enabling automated conversions from frames to stubs/tracks and vice versa
In data members of classes Stub* & Track* below, the variables describing
stubs/tracks are stored both in digitial format as a 64b word in frame_,
and in undigitized format in an std::tuple. (This saves CPU)
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <vector>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <iostream>
#include <string>

namespace trklet {

  // hybrid processes
  enum class Process { begin, tm = begin, dr, kf, tfp, end, x };
  // hybrid variables
  enum class Variable { begin, stubId = begin, r, phi, z, dPhi, dZ, inv2R, phiT, cot, zT, end, x };
  // hybrid process order
  constexpr std::initializer_list<Process> Processes = {Process::tm, Process::dr, Process::kf, Process::tfp};
  // conversion: Process to int
  inline constexpr int operator+(Process p) { return static_cast<int>(p); }
  // conversion: Variable to int
  inline constexpr int operator+(Variable v) { return static_cast<int>(v); }
  // increment of Process
  inline constexpr Process operator++(Process p) { return Process(+p + 1); }
  // increment of Variable
  inline constexpr Variable operator++(Variable v) { return Variable(+v + 1); }

  //Base class representing format of a variable
  class DataFormat {
  public:
    DataFormat(bool twos, bool biased = true) : twos_(twos), width_(0), base_(1.), range_(0.) {}
    DataFormat(bool twos, int width, double base, double range)
        : twos_(twos), width_(width), base_(base), range_(range) {}
    DataFormat() {}
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

  // class representing format of a specific variable
  template <Variable v, Process p>
  class Format : public DataFormat {
  public:
    Format(const ChannelAssignment* ca);
    ~Format() {}
  };

  template <>
  Format<Variable::inv2R, Process::tfp>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::phiT, Process::tfp>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::cot, Process::tfp>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::zT, Process::tfp>::Format(const ChannelAssignment* ca);

  template <>
  Format<Variable::inv2R, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::phiT, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::zT, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::cot, Process::tm>::Format(const ChannelAssignment* ca);

  template <>
  Format<Variable::stubId, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::r, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::phi, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::z, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::dPhi, Process::tm>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::dZ, Process::tm>::Format(const ChannelAssignment* ca);

  template <>
  Format<Variable::inv2R, Process::kf>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::phiT, Process::kf>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::cot, Process::kf>::Format(const ChannelAssignment* ca);
  template <>
  Format<Variable::zT, Process::kf>::Format(const ChannelAssignment* ca);

  /*! \class  trklet::DataFormats
   *  \brief  Class to calculate and provide dataformats used by Hybrid emulator
   *  \author Thomas Schuh
   *  \date   2024, Sep
   */
  class DataFormats {
  private:
    // variable flavour mapping, Each row below declares which processing steps use the variable named in the comment at the end of the row
    static constexpr std::array<std::array<Process, +Process::end>, +Variable::end> config_ = {{
        //  Process::tm  Process::dr   Process::kf   Process::tfp
        {{Process::tm, Process::x, Process::x, Process::x}},      // Variable::stubId
        {{Process::tm, Process::tm, Process::tm, Process::x}},    // Variable::r
        {{Process::tm, Process::tm, Process::tm, Process::x}},    // Variable::phi
        {{Process::tm, Process::tm, Process::tm, Process::x}},    // Variable::z
        {{Process::tm, Process::tm, Process::tm, Process::x}},    // Variable::dPhi
        {{Process::tm, Process::tm, Process::tm, Process::x}},    // Variable::dZ
        {{Process::tm, Process::tm, Process::kf, Process::tfp}},  // Variable::inv2R
        {{Process::tm, Process::tm, Process::kf, Process::tfp}},  // Variable::phiT
        {{Process::tm, Process::tm, Process::kf, Process::tfp}},  // Variable::cot
        {{Process::tm, Process::tm, Process::kf, Process::tfp}}   // Variable::zT
    }};
    // stub word assembly, shows which stub variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> stubs_ = {{
        {Variable::stubId, Variable::r, Variable::phi, Variable::z},              // Process::tm
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},  // Process::dr
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},  // Process::kf
        {}                                                                        // Process::tfp
    }};
    // track word assembly, shows which track variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> tracks_ = {{
        {Variable::inv2R, Variable::phiT, Variable::zT},                 // Process::tm
        {Variable::inv2R, Variable::phiT, Variable::zT},                 // Process::dr
        {Variable::inv2R, Variable::phiT, Variable::cot, Variable::zT},  // Process::kf
        {}                                                               // Process::tfp
    }};

  public:
    DataFormats();
    DataFormats(const ChannelAssignment* ca);
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
    const tt::Setup* setup() const { return channelAssignment_->setup(); }
    // number of bits being used for specific variable flavour
    int width(Variable v, Process p) const { return formats_[+v][+p]->width(); }
    // precision being used for specific variable flavour
    double base(Variable v, Process p) const { return formats_[+v][+p]->base(); }
    // covered range for specific variable flavour
    double range(Variable v, Process p) const { return formats_[+v][+p]->range(); }
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
    const ChannelAssignment* channelAssignment_;
    // collection of unique formats
    std::vector<DataFormat> dataFormats_;
    // variable flavour mapping
    std::vector<std::vector<DataFormat*>> formats_;
    // number of unused frame bits for a all Stub flavours
    std::vector<int> numUnusedBitsStubs_;
    // number of unused frame bits for a all Track flavours
    std::vector<int> numUnusedBitsTracks_;
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
        : dataFormats_(stub.dataFormats()), p_(++stub.p()), frame_(stub.frame()), data_(data...) {
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

  // class to represent stubs generated by process TrackMulitplexer
  class StubTM : public Stub<int, double, double, double> {
  public:
    // construct StubTM from Frame
    StubTM(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::tm) {}
    // construct StubTM from TTStubRef
    StubTM(const TTStubRef& ttStubRef, const DataFormats* df, int stubId, double r, double phi, double z)
        : Stub(ttStubRef, df, Process::tm, stubId, r, phi, z) {}
    ~StubTM() {}
    // stub Id
    int stubId() const { return std::get<0>(data_); }
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<1>(data_); }
    // stub phi wrt processing nonant centre
    double phi() const { return std::get<2>(data_); }
    // stub z
    double z() const { return std::get<3>(data_); }
  };

  // class to represent stubs generated by process DuplicateRemoval
  class StubDR : public Stub<double, double, double, double, double> {
  public:
    // construct StubDR from Frame
    StubDR(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::dr) {}
    // construct StubDR from StubTM
    StubDR(const StubTM& stub, double r, double phi, double z, double dPhi, double dZ)
        : Stub(stub, r, phi, z, dPhi, dZ) {}
    ~StubDR() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt phi sector centre
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); }
    // stub phi uncertainty
    double dPhi() const { return std::get<3>(data_); }
    // stub z uncertainty
    double dZ() const { return std::get<4>(data_); }
  };

  // class to represent stubs generated by process KalmanFilter
  class StubKF : public Stub<double, double, double, double, double> {
  public:
    // construct StubKF from Frame
    StubKF(const tt::FrameStub& fs, const DataFormats* df) : Stub(fs, df, Process::kf) {}
    // construct StubKF from StubDR
    StubKF(const StubDR& stub, double r, double phi, double z, double dPhi, double dZ)
        : Stub(stub, r, phi, z, dPhi, dZ) {}
    ~StubKF() {}
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); };
    // stub phi residual wrt track parameter
    double phi() const { return std::get<1>(data_); };
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); };
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
        : dataFormats_(track.dataFormats()), p_(++track.p()), frame_(track.frame()), data_(data...) {
      dataFormats_->convertTrack(p_, data_, frame_.second);
    }
    Track() {}
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

  // class to represent tracks generated by process TrackMultiplexer
  class TrackTM : public Track<double, double, double> {
  public:
    // construct TrackTM from Frame
    TrackTM(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::tm) {}
    // construct TrackTM from TTTrack
    TrackTM(const TTTrackRef& tTTrackRef, const DataFormats* df, double inv2R, double phiT, double zT)
        : Track(tTTrackRef, df, Process::tm, inv2R, phiT, zT) {}
    ~TrackTM() {}
    // track inv2R
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius chosenRofPhi wrt pprocessing centre
    double phiT() const { return std::get<1>(data_); }
    // track z at radius chosenRofZ
    double zT() const { return std::get<2>(data_); }
  };

  // class to represent tracks generated by process DuplicateRemoval
  class TrackDR : public Track<double, double, double> {
  public:
    // construct TrackDR from Frame
    TrackDR(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::dr) {}
    // construct TrackDR from TrackTM
    TrackDR(const TrackTM& track) : Track(track, track.inv2R(), track.phiT(), track.zT()) {}
    ~TrackDR() {}
    // track qOver pt
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius chosenRofPhi wrt processing nonant centre
    double phiT() const { return std::get<1>(data_); }
    // track z at radius chosenRofZ
    double zT() const { return std::get<2>(data_); }
  };

  // class to represent tracks generated by process KalmanFilter
  class TrackKF : public Track<double, double, double, double> {
  public:
    // construct TrackKF from Frame
    TrackKF(const tt::FrameTrack& ft, const DataFormats* df) : Track(ft, df, Process::kf) {}
    // construct TrackKF from TrackDR
    TrackKF(const TrackDR& track, double inv2R, double phiT, double cot, double zT)
        : Track(track, inv2R, phiT, cot, zT) {}
    TrackKF() {}
    ~TrackKF() {}
    // track inv2R
    double inv2R() const { return std::get<0>(data_); }
    // track phi at radius 0 wrt processing nonant centre
    double phiT() const { return std::get<1>(data_); }
    // track cotThea
    double cot() const { return std::get<2>(data_); }
    // track z at radius 0
    double zT() const { return std::get<3>(data_); }
  };

}  // namespace trklet

EVENTSETUP_DATA_DEFAULT_RECORD(trklet::DataFormats, trklet::ChannelAssignmentRcd);

#endif
