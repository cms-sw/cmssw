#ifndef L1Trigger_TrackFindingTracklet_DR_h
#define L1Trigger_TrackFindingTracklet_DR_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <vector>

namespace trklet {

  /*! \class  trklet::DR
   *  \brief  Class to bit- and clock-accurate emulate duplicate removal
   *          DR identifies duplicates based on pairs of tracks that share stubs in at least 3 layers.
   *          It keeps the first such track in each pair.
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class DR {
  public:
    DR(const edm::ParameterSet& iConfig,
       const tt::Setup* setup_,
       const trackerTFP::DataFormats* dataFormats,
       const ChannelAssignment* channelAssignment,
       int region);
    ~DR() {}
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub);
    // fill output products
    void produce(tt::StreamsStub& accpetedStubs,
                 tt::StreamsTrack& acceptedTracks,
                 tt::StreamsStub& lostStubs,
                 tt::StreamsTrack& lostTracks);

  private:
    struct Stub {
      Stub(const tt::FrameStub& frame, int stubId, int channel) : frame_(frame), stubId_(stubId), channel_(channel) {}
      bool operator==(const Stub& s) const { return s.stubId_ == stubId_; }
      tt::FrameStub frame_;
      // all stubs id
      int stubId_;
      // kf layer id
      int channel_;
    };
    struct Track {
      // max number of stubs a track may formed of (we allow only one stub per layer)
      static constexpr int max_ = 7;
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
    // provides dataformats
    const trackerTFP::DataFormats* dataFormats_;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_;
    // processing region (0 - 8) aka processing phi nonant
    const int region_;
    // storage of input tracks
    std::vector<Track> tracks_;
    // storage of input stubs
    std::vector<Stub> stubs_;
    // h/w liked organized pointer to input tracks
    std::vector<std::vector<Track*>> input_;
  };

}  // namespace trklet

#endif
