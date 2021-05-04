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

namespace trklet {

  class L1SimTrack {
  public:
    L1SimTrack();
    L1SimTrack(int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz);
    ~L1SimTrack() = default;

    void write(std::ofstream& out);
    void write(std::ostream& out);

    int eventid() const { return eventid_; }
    int trackid() const { return trackid_; }
    int type() const { return type_; }
    double pt() const { return pt_; }
    double eta() const { return eta_; }
    double phi() const { return phi_; }
    double vx() const { return vx_; }
    double vy() const { return vy_; }
    double vz() const { return vz_; }
    double dxy() const { return -vx() * sin(phi()) + vy() * cos(phi()); }
    double d0() const { return -dxy(); }
    int charge() const {
      if (type_ == 11 || type_ == 13 || type_ == -211 || type_ == -321 || type_ == -2212)
        return -1;
      return 1;
    }

  private:
    int eventid_;
    int trackid_;
    int type_;
    double pt_;
    double eta_;
    double phi_;
    double vx_;
    double vy_;
    double vz_;
  };

  class SLHCEvent {
  public:
    SLHCEvent() {
      //empty constructor to be used with 'filler' functions
      eventnum_ = 0;
    }
    SLHCEvent(std::istream& in);
    ~SLHCEvent() = default;

    void setIPx(double x) { x_offset_ = x; }
    void setIPy(double y) { y_offset_ = y; }

    void setEventNum(int eventnum) { eventnum_ = eventnum; }

    void addL1SimTrack(
        int eventid, int trackid, int type, double pt, double eta, double phi, double vx, double vy, double vz);

    bool addStub(int layer,
                 int ladder,
                 int module,
                 int strip,
                 int eventid,
                 std::vector<int> tps,
                 double pt,
                 double bend,
                 double x,
                 double y,
                 double z,
                 int isPSmodule,
                 int isFlipped);

    const L1TStub& lastStub() const { return stubs_.back(); }

    void write(std::ofstream& out);
    void write(std::ostream& out);

    unsigned int layersHit(int tpid, int& nlayers, int& ndisks);

    int nstubs() const { return stubs_.size(); }

    const L1TStub& stub(int i) const { return stubs_[i]; }

    unsigned int nsimtracks() const { return simtracks_.size(); }

    const L1SimTrack& simtrack(int i) const { return simtracks_[i]; }

    int eventnum() const { return eventnum_; }

    int getSimtrackFromSimtrackid(int simtrackid, int eventid = 0) const;

  private:
    int eventnum_;
    std::vector<L1SimTrack> simtracks_;
    std::vector<L1TStub> stubs_;

    double x_offset_{0.0};
    double y_offset_{0.0};
  };

};  // namespace trklet
#endif
