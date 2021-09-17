#ifndef L1Trigger_TrackFindingTracklet_interface_FullMatchMemory_h
#define L1Trigger_TrackFindingTracklet_interface_FullMatchMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>
#include <string>
#include <utility>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class Tracklet;

  class FullMatchMemory : public MemoryBase {
  public:
    FullMatchMemory(std::string name, Settings const& settings);

    ~FullMatchMemory() override = default;

    void addMatch(Tracklet* tracklet, const Stub* stub);

    unsigned int nMatches() const { return matches_.size(); }

    Tracklet* getTracklet(unsigned int i) { return matches_[i].first; }

    std::pair<Tracklet*, const Stub*> getMatch(unsigned int i) { return matches_[i]; }

    void clean() override { matches_.clear(); }

    void writeMC(bool first, unsigned int iSector);

    int layer() const { return layer_; }
    int disk() const { return disk_; }

  private:
    std::vector<std::pair<Tracklet*, const Stub*> > matches_;

    int layer_;
    int disk_;
  };

};  // namespace trklet
#endif
