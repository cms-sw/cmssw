// This holds two classes: L1SimTrack (truth level simulated track), and SLHCEvent (support for maintaining standalone running)
#ifndef L1Trigger_TrackFindingTracklet_interface_SLHCEvent_h
#define L1Trigger_TrackFindingTracklet_interface_SLHCEvent_h

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1SimTrack.h"

namespace trklet {

  class SLHCEvent {
  public:
    SLHCEvent() {
      //empty constructor to be used with 'filler' functions
      eventnum_ = 0;
    }
    SLHCEvent(std::istream& in);
    ~SLHCEvent() = default;

    void setEventNum(int eventnum) { eventnum_ = eventnum; }

    void addL1SimTrack(
        int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz);

    bool addStub(std::string DTClink,
                 int region,
                 int layerdisk,
                 std::string stubword,
                 int isPSmodule,
                 int isFlipped,
                 double x,
                 double y,
                 double z,
                 double bend,
                 double strip,
                 std::vector<int> tps);

    const L1TStub& lastStub() const { return stubs_.back(); }

    void setIP(double x, double y) {
      ipx_ = x;
      ipy_ = y;
    }

    void write(std::ofstream& out);

    unsigned int layersHit(int tpid, int& nlayers, int& ndisks);

    int nstubs() const { return stubs_.size(); }

    const L1TStub& stub(int i) const { return stubs_[i]; }

    unsigned int nsimtracks() const { return simtracks_.size(); }

    const L1SimTrack& simtrack(int i) const { return simtracks_[i]; }

    int eventnum() const { return eventnum_; }

  private:
    int eventnum_;
    std::vector<L1SimTrack> simtracks_;
    std::vector<L1TStub> stubs_;
    double ipx_, ipy_;
  };

};  // namespace trklet
#endif
