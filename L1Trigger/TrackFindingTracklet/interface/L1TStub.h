#ifndef L1Trigger_TrackFindingTracklet_interface_L1TStub_h
#define L1Trigger_TrackFindingTracklet_interface_L1TStub_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

namespace trklet {

  class L1TStub {
  public:
    L1TStub();

    L1TStub(std::string DTClink,
            int region,
            int layerdisk,
            std::string stubword,
            int isPSmodule,
            int isFlipped,
            bool tiltedBarrel,
            unsigned int tiltedRingId,
            unsigned int endcapRingId,
            unsigned int detId,
            double x,
            double y,
            double z,
            double bend,
            double strip,
            std::vector<int> tps);

    ~L1TStub() = default;

    void write(std::ofstream& out);

    double diphi();

    double iphiouter();

    double diz();

    unsigned int layer() const { return layer_; }
    int disk() const {
      if (layerdisk_ < N_LAYER) {
        return 0;
      }
      int disk = layerdisk_ - N_LAYER + 1;
      if (z_ < 0.0) {
        return -disk;
      }
      return disk;
    }

    double x() const { return x_; }
    double y() const { return y_; }
    double z() const { return z_; }
    double r() const { return std::hypot(x_, y_); }
    double pt() const { return pt_; }
    double r2() const { return x_ * x_ + y_ * y_; }
    double bend() const { return bend_; }

    double phi() const { return atan2(y_, x_); }

    unsigned int iphi() const { return iphi_; }
    unsigned int iz() const { return iz_; }

    void setiphi(int iphi) { iphi_ = iphi; }
    void setiz(int iz) { iz_ = iz; }

    double sigmax() const { return sigmax_; }
    double sigmaz() const { return sigmaz_; }

    bool operator==(const L1TStub& other) const;

    void lorentzcor(double shift);

    int eventid() const { return eventid_; }
    std::vector<int> tps() const { return tps_; }

    void setAllStubIndex(unsigned int index) { allstubindex_ = index; }
    void setUniqueIndex(unsigned int index) { uniqueindex_ = index; }
    void setCoords(double x, double y, double z) {
      x_ = x;
      y_ = y;
      z_ = z;
    }

    unsigned int allStubIndex() const { return allstubindex_; }
    unsigned int uniqueIndex() const { return uniqueindex_; }

    unsigned int strip() const { return strip_; }

    double alpha(double pitch) const;

    //Scaled to go between -1 and +1
    double alphanorm() const;

    void setXY(double x, double y);

    unsigned int isPSmodule() const { return isPSmodule_; }
    unsigned int isFlipped() const { return isFlipped_; }

    bool isTilted() const { return tiltedBarrel_; }  // Tilted barrel

    // Tilted module ring no. (Increasing 1 to 12 as |z| increases).
    unsigned int tiltedRingId() const { return tiltedRingId_; }
    // Endcap disk module ring number (1-15 as r increases).
    unsigned int endcapRingId() const { return endcapRingId_; }
    unsigned int detId() const { return detId_; }  // Of lower sensor in module

    bool tpmatch(int tp) const;
    bool tpmatch2(int tp) const;

    const std::string& DTClink() const { return DTClink_; }

    int layerdisk() const { return layerdisk_; }

    int region() const { return region_; }

    const std::string& stubword() const { return stubword_; }

  private:
    int layerdisk_;
    std::string DTClink_;
    int region_;
    std::string stubword_;
    int eventid_;
    std::vector<int> tps_;
    unsigned int iphi_;
    unsigned int iz_;
    unsigned int layer_;
    unsigned int strip_;
    double x_;
    double y_;
    double z_;
    double sigmax_;
    double sigmaz_;
    double pt_;
    double bend_;
    unsigned int allstubindex_;
    unsigned int uniqueindex_;
    unsigned int isPSmodule_;
    unsigned int isFlipped_;
    bool tiltedBarrel_;
    unsigned int tiltedRingId_;
    unsigned int endcapRingId_;
    unsigned int detId_;
  };
};  // namespace trklet
#endif
