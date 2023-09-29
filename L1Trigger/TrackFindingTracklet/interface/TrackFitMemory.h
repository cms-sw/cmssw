#ifndef L1Trigger_TrackFindingTracklet_interface_TrackFitMemory_h
#define L1Trigger_TrackFindingTracklet_interface_TrackFitMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class Tracklet;

  class TrackFitMemory : public MemoryBase {
  public:
    TrackFitMemory(std::string name, Settings const& settings, double phimin, double phimax);

    ~TrackFitMemory() override = default;

    void addTrack(Tracklet* tracklet) { tracks_.push_back(tracklet); }
    void addStubList(std::vector<const Stub*> stublist) { stublists_.push_back(stublist); }
    void addStubidsList(std::vector<std::pair<int, int>> stubidslist) { stubidslists_.push_back(stubidslist); }

    unsigned int nTracks() const { return tracks_.size(); }
    unsigned int nStublists() const { return stublists_.size(); }
    unsigned int nStubidslists() const { return stubidslists_.size(); }

    Tracklet* getTrack(unsigned int i) { return tracks_[i]; }
    // Get pointers to Stubs on track.
    std::vector<const Stub*> getStublist(unsigned int i) const { return stublists_[i]; }
    // Get (layer, unique stub index in layer) of stubs on track.
    std::vector<std::pair<int, int>> getStubidslist(unsigned int i) const { return stubidslists_[i]; }

    void clean() override {
      tracks_.clear();
      stublists_.clear();
      stubidslists_.clear();
    }

    void writeTF(bool first, unsigned int iSector);

  private:
    double phimin_;
    double phimax_;
    std::vector<Tracklet*> tracks_;
    std::vector<std::vector<const Stub*>> stublists_;
    std::vector<std::vector<std::pair<int, int>>> stubidslists_;
  };

};  // namespace trklet
#endif
