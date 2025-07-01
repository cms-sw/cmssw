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

    void addProj(Tracklet* tracklet, unsigned int page = 0);

    unsigned int nTracklets(unsigned int page = 0) const { return tracklets_[page].size(); }

    Tracklet* getTracklet(unsigned int i, unsigned int page = 0) { return tracklets_[page][i]; }

    unsigned int nPage() const { return npage_; }

    void clean() override;

    void writeTPROJ(bool first, unsigned int iSector);

    int layer() const { return layer_; }
    int disk() const { return disk_; }

  private:
    std::vector<std::vector<Tracklet*> > tracklets_;

    bool hasProj_;
    int layer_;
    int disk_;
    int npage_;
  };

};  // namespace trklet
#endif
