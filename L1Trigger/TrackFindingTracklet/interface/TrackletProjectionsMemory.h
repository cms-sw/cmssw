// TrackletProjectionsMemory: this class holds the
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletProjectionsMemory_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletProjectionsMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Tracklet;

  class TrackletProjectionsMemory : public MemoryBase {
  public:
    TrackletProjectionsMemory(std::string name, Settings const& settings);

    ~TrackletProjectionsMemory() override {
      if (settings_.writeMonitorData("WriteEmptyProj") && (!hasProj_)) {
        edm::LogPrint("Tracklet") << "Empty Projection Memory : " << getName() << std::endl;
      }
    };

    void addProj(Tracklet* tracklet);

    unsigned int nTracklets() const { return tracklets_.size(); }

    Tracklet* getTracklet(unsigned int i) { return tracklets_[i]; }

    void clean() override;

    void writeTPROJ(bool first, unsigned int iSector);

    int layer() const { return layer_; }
    int disk() const { return disk_; }

  private:
    std::vector<Tracklet*> tracklets_;

    bool hasProj_;
    int layer_;
    int disk_;
  };

};  // namespace trklet
#endif
