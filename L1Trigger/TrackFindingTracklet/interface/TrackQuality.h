#ifndef L1Trigger_TrackFindingTracklet_TrackQuality_h
#define L1Trigger_TrackFindingTracklet_TrackQuality_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"

#include <vector>
#include <array>
#include "conifer.h"
#include "ap_fixed.h"

namespace trklet {

  typedef ap_int<20> AP_INT_BDT;
  typedef ap_fixed<20, 10> AP_FIXED_BDT;
  typedef conifer::BDT<AP_FIXED_BDT, AP_FIXED_BDT> EmulatorBDT;

  /*! \class  trklet::TrackQuality
   *  \brief  Bit accurate emulation of the track quality BDT including calculation of chi2s.
   *  \author Thomas Schuh
   *  \date   Aug 2025
   */
  class TrackQuality {
  public:
    // various internal dataformats
    struct InternalFormats {
      DataFormat m02_;
      DataFormat m12_;
      DataFormat invV0_;
      DataFormat invV1_;
    };
    TrackQuality(const DataFormats* df, const InternalFormats& internal, int region, const EmulatorBDT* bdt)
        : setup_(df->setup()),
          channelAssignment_(df->channelAssignment()),
          dataFormats_(df),
          internalFormats_(&internal),
          region_(region),
          bdt_(bdt) {}
    ~TrackQuality() = default;
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack&, const tt::StreamsStub&);
    // fills output products
    void produce(tt::StreamsTrack&) const;

  private:
    // representation of an input Frame
    struct Frame {
      Frame() : track_(nullptr), stubs_() {}
      Frame(TrackKF* track, int n) : track_(track), stubs_(n, nullptr) {}
      TrackKF* track_;
      std::vector<StubKF*> stubs_;
    };
    // helper class to store configurations
    const tt::Setup* setup_;
    // helper class to store configurations
    const ChannelAssignment* channelAssignment_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // collection of internal formats
    const InternalFormats* internalFormats_;
    // region id
    int region_;
    // KF tracks
    std::vector<TrackKF> tracks_;
    // KF stubs
    std::vector<StubKF> stubs_;
    // input data
    std::vector<Frame> input_;
    // bdt model
    const EmulatorBDT* bdt_;
    // copy of input track streams
    tt::StreamsTrack streams_;
  };

}  // namespace trklet

#endif
