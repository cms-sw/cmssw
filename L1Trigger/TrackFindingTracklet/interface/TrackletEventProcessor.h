// TrackletEventProcessor: Class responsible for the main event processing for the tracklet algorithm
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletEventProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletEventProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/Timer.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <map>
#include <memory>
#include <vector>
#include <deque>
#include <string>

namespace trklet {

  class Settings;
  class SLHCEvent;
  class Globals;
  class Sector;
  class HistBase;
  class Track;

  class TrackletEventProcessor {
  public:
    TrackletEventProcessor();

    ~TrackletEventProcessor();

    void init(Settings const& theSettings, const ChannelAssignment* channelAssignment);

    void event(SLHCEvent& ev);

    void printSummary();

    const std::vector<Track>& tracks() const { return tracks_; }

    void produce(tt::Streams& streamsTrack, tt::StreamsStub& streamsStub);

  private:
    void configure(std::istream& inwire, std::istream& inmem, std::istream& inproc);

    const Settings* settings_{nullptr};
    const ChannelAssignment* channelAssignment_{nullptr};

    std::unique_ptr<Globals> globals_;

    std::unique_ptr<Sector> sector_;

    HistBase* histbase_{};

    int eventnum_ = {0};

    Timer cleanTimer_;
    Timer addStubTimer_;
    Timer InputRouterTimer_;
    Timer VMRouterTimer_;
    Timer TETimer_;
    Timer TEDTimer_;
    Timer TRETimer_;
    Timer TPTimer_;
    Timer TCTimer_;
    Timer TCDTimer_;
    Timer PRTimer_;
    Timer METimer_;
    Timer MCTimer_;
    Timer MPTimer_;
    Timer FTTimer_;
    Timer PDTimer_;

    std::vector<Track> tracks_;
    tt::Streams streamsTrack_;
    tt::StreamsStub streamsStub_;
  };

};  // namespace trklet
#endif
