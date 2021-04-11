#ifndef L1Trigger_TrackFindingTracklet_interface_AllProjectionsMemory_h
#define L1Trigger_TrackFindingTracklet_interface_AllProjectionsMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Tracklet;

  class AllProjectionsMemory : public MemoryBase {
  public:
    AllProjectionsMemory(std::string name, Settings const& settings);

    ~AllProjectionsMemory() override = default;

    void addTracklet(Tracklet* tracklet) { tracklets_.push_back(tracklet); }

    unsigned int nTracklets() const { return tracklets_.size(); }

    const Tracklet* getTracklet(unsigned int i) const { return tracklets_[i]; }

    void clean() override { tracklets_.clear(); }

    void writeAP(bool first, unsigned int iSector);

  private:
    std::vector<Tracklet*> tracklets_;

    int layer_;
    int disk_;
  };
};  // namespace trklet
#endif
