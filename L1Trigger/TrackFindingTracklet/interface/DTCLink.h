// This class holds a list of stubs that are in a given layer and DCT region
#ifndef L1Trigger_TrackFindingTracklet_interface_DTCLink_h
#define L1Trigger_TrackFindingTracklet_interface_DTCLink_h

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"

namespace trklet {

  class DTCLink {
  public:
    DTCLink(double phimin, double phimax);

    ~DTCLink() = default;

    void addStub(std::pair<Stub*, L1TStub*> stub);

    bool inRange(double phi, bool overlaplayer);

    unsigned int nStubs() const { return stubs_.size(); }

    Stub* getFPGAStub(unsigned int i) const { return stubs_[i].first; }
    L1TStub* getL1TStub(unsigned int i) const { return stubs_[i].second; }
    std::pair<Stub*, L1TStub*> getStub(unsigned int i) const { return stubs_[i]; }

    void clean() { stubs_.clear(); }

  private:
    double phimin_;
    double phimax_;
    std::vector<std::pair<Stub*, L1TStub*> > stubs_;
  };
};  // namespace trklet
#endif
