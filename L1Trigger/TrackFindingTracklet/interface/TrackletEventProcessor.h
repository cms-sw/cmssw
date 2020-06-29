// TrackletEventProcessor: Class responsible for the main event processing for the tracklet algorithm
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletEventProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletEventProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/Timer.h"

#include <map>
#include <memory>
#include <vector>
#include <string>

namespace trklet {

  class Settings;
  class SLHCEvent;
  class Globals;
  class Sector;
  class HistBase;
  class Track;
  class Cabling;

  class TrackletEventProcessor {
  public:
    TrackletEventProcessor();

    ~TrackletEventProcessor();

    void init(const Settings* theSettings);

    void event(SLHCEvent& ev);

    void printSummary();

    std::vector<Track*>& tracks() { return tracks_; }

  private:
    const Settings* settings_{nullptr};

    std::unique_ptr<Globals> globals_;

    std::vector<std::unique_ptr<Sector> > sectors_;

    HistBase* histbase_{};

    int eventnum_ = {0};

    std::unique_ptr<Cabling> cabling_;

    Timer cleanTimer_;
    Timer addStubTimer_;
    Timer VMRouterTimer_;
    Timer TETimer_;
    Timer TEDTimer_;
    Timer TRETimer_;
    Timer TCTimer_;
    Timer TCDTimer_;
    Timer PRTimer_;
    Timer METimer_;
    Timer MCTimer_;
    Timer MPTimer_;
    Timer FTTimer_;
    Timer PDTimer_;

    std::vector<Track*> tracks_;

    std::map<std::string, std::vector<int> > dtclayerdisk_;
  };

};  // namespace trklet
#endif
