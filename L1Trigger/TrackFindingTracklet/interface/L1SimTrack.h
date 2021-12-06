// Stores MC truth information, pt, eta, phi, vx, vy, vz, as well as particle type and track id
#ifndef L1Trigger_TrackFindingTracklet_interface_L1SimTrack_h
#define L1Trigger_TrackFindingTracklet_interface_L1SimTrack_h

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>

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

};  // namespace trklet
#endif
