#ifndef L1Trigger_TrackFindingTracklet_TrackMultiplexer_h
#define L1Trigger_TrackFindingTracklet_TrackMultiplexer_h

#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"

#include <vector>
#include <deque>

namespace trklet {

  /*! \class  trklet::TrackMultiplexer
   *  \brief  Class to emulate transformation of tracklet tracks and stubs into TMTT format
   *          and routing of seed type streams into single stream
   *  \author Thomas Schuh
   *  \date   2023, Jan
   */
  class TrackMultiplexer {
  public:
    TrackMultiplexer(const Setup*, int);
    ~TrackMultiplexer() = default;
    // read in and organize input tracks and stubs
    void consume(const tt::StreamsTrack&, const tt::StreamsStub&);
    // fill output products
    void produce(tt::StreamsTrack&, tt::StreamsStub&);

  private:
    struct Track {
      Track(int channel, int frame) : channel_(channel), frame_(frame) {}
      int channel_;
      int frame_;
    };
    // remove and return first element of deque, returns nullptr if empty
    template <class T>
    T* pop_front(std::deque<T*>& ts) const;
    // true if truncation is enbaled
    bool enableTruncation_;
    // stub residuals are recalculated from seed parameter and TTStub position
    bool useTTStubResiduals_;
    // track parameter are recalculated from seed TTStub positions
    bool useTTStubParameters_;
    // provides run-time constants
    const Setup* setup_;
    // processing region (0 - 8) aka processing phi nonant
    const int region_;
    // input track streams
    tt::StreamsTrack streamsTrack_;
    // input stub streams
    tt::StreamsStub streamsStub_;
    // storage of input tracks
    std::vector<Track> tracks_;
    // h/w liked organized pointer to input tracks
    std::vector<std::vector<Track*>> input_;
  };

}  // namespace trklet

#endif
