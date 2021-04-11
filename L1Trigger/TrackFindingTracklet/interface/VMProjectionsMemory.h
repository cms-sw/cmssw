// VMProjectionsMemory: Class to hold a reduced format of the tracklet projections (from ProjectionRouter)
#ifndef L1Trigger_TrackFindingTracklet_interface_VMProjectionsMemory_h
#define L1Trigger_TrackFindingTracklet_interface_VMProjectionsMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>

namespace trklet {

  class Settings;
  class Tracklet;

  class VMProjectionsMemory : public MemoryBase {
  public:
    VMProjectionsMemory(std::string name, Settings const& settings);

    ~VMProjectionsMemory() override = default;

    void addTracklet(Tracklet* tracklet, unsigned int allprojindex);

    unsigned int nTracklets() const { return tracklets_.size(); }

    Tracklet* getTracklet(unsigned int i) { return tracklets_[i].first; }
    int getAllProjIndex(unsigned int i) const { return tracklets_[i].second; }

    void writeVMPROJ(bool first, unsigned int iSector);

    void clean() override { tracklets_.clear(); }

    int layer() const { return layer_; }
    int disk() const { return disk_; }

  private:
    int layer_;
    int disk_;
    std::vector<std::pair<Tracklet*, unsigned int> > tracklets_;
  };

};  // namespace trklet
#endif
