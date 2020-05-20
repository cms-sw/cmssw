#ifndef L1Trigger_TrackFindingTracklet_interface_DTC_h
#define L1Trigger_TrackFindingTracklet_interface_DTC_h

#include "L1Trigger/TrackFindingTracklet/interface/DTCLink.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

namespace trklet {

  class Stub;
  class L1TStub;

  class DTC {
  public:
    DTC(std::string name = "");

    ~DTC() = default;

    void setName(std::string name);

    void addSec(int sector);

    void addphi(double phi, unsigned int layerdisk);

    void addLink(double phimin, double phimax);

    int addStub(std::pair<Stub*, L1TStub*> stub);

    unsigned int nLinks() const { return links_.size(); }

    const DTCLink& link(unsigned int i) const { return links_[i]; }

    void clean();

    double min(unsigned int i) const { return phimin_[i]; }
    double max(unsigned int i) const { return phimax_[i]; }

  private:
    std::string name_;
    std::vector<DTCLink> links_;
    std::vector<int> sectors_;

    double phimin_[N_LAYERDISK];
    double phimax_[N_LAYERDISK];
  };
};  // namespace trklet
#endif
