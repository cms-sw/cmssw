#ifndef L1Trigger_TrackFindingTracklet_DuplicateRemoval_h
#define L1Trigger_TrackFindingTracklet_DuplicateRemoval_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <vector>

namespace trklet {

  /*! \class  trklet::DuplicateRemoval
   *  \brief  Class to bit- and clock-accurate emulate duplicate removal
   *          DR identifies duplicates based on pairs of tracks that share stubs in at least 3 layers.
   *          It keeps the first such track in each pair. The Track order is determined by TrackMultiplexer,
   *          provided by ProducerTM.
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class DuplicateRemoval {
  public:
    DuplicateRemoval(const edm::ParameterSet& iConfig,
                     const tt::Setup* setup_,
                     const trackerTFP::LayerEncoding* layerEncoding,
                     const DataFormats* dataFormats,
                     const ChannelAssignment* channelAssignment,
                     int region);
    ~DuplicateRemoval() {}
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub);
    // fill output products
    void produce(tt::StreamsTrack& acceptedTracks, tt::StreamsStub& accpetedStubs);

  private:
    struct Stub {
      Stub(const tt::FrameStub& frame, int stubId, int layer) : frame_(frame), stubId_(stubId), layer_(layer) {}
      // output frame
      tt::FrameStub frame_;
      // all stubs id
      int stubId_;
      // kf layer id
      int layer_;
    };
    struct Track {
      // max number of stubs a track may formed of (we allow only one stub per layer)
      static constexpr int max_ = 11;
      Track() { stubs_.reserve(max_); }
      Track(const tt::FrameTrack& frame, const std::vector<Stub*>& stubs) : frame_(frame), stubs_(stubs) {}
      tt::FrameTrack frame_;
      std::vector<Stub*> stubs_;
    };
    // compares two tracks, returns true if those are considered duplicates
    bool equalEnough(Track* t0, Track* t1) const;
    // true if truncation is enbaled
    bool enableTruncation_;
    // provides run-time constants
    const tt::Setup* setup_;
    // helper class to encode layer
    const trackerTFP::LayerEncoding* layerEncoding_;
    // provides dataformats
    const DataFormats* dataFormats_;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_;
    // processing region (0 - 8) aka processing phi nonant
    const int region_;
    // storage of input tracks
    std::vector<Track> tracks_;
    // storage of input stubs
    std::vector<Stub> stubs_;
    // h/w liked organized pointer to input tracks
    std::vector<Track*> input_;
    // dataformat used to calculate pitch over stubs radius
    DataFormat r_;
  };

}  // namespace trklet

#endif
