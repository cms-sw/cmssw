#ifndef L1Trigger_TrackFindingTracklet_interface_CleanTrackMemory_h
#define L1Trigger_TrackFindingTracklet_interface_CleanTrackMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>

namespace trklet {

  class Settings;
  class Tracklet;

  class CleanTrackMemory : public MemoryBase {
  public:
    CleanTrackMemory(std::string name, Settings const& settings, double phimin, double phimax);

    ~CleanTrackMemory() override = default;

    void addTrack(Tracklet* tracklet) { tracks_.push_back(tracklet); }

    unsigned int nTracks() const { return tracks_.size(); }

    void clean() override { tracks_.clear(); }

    void writeCT(bool first, unsigned int iSector);

  private:
    double phimin_;
    double phimax_;
    std::vector<Tracklet*> tracks_;
  };

};  // namespace trklet
#endif
