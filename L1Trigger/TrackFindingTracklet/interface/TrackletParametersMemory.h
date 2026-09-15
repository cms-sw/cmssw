// TrackletParametersMemory: This class holds the tracklet parameters for selected stub pairs.
// This class owns the tracklets. Further modules only holds pointers.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletParametersMemory_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletParametersMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class Tracklet;

  class TrackletParametersMemory : public MemoryBase {
  public:
    TrackletParametersMemory(std::string name, Settings const &settings);

    ~TrackletParametersMemory() override = default;

    void addTracklet(Tracklet *tracklet, unsigned int page = 0) { tracklets_[page].push_back(tracklet); }

    unsigned int nTracklets(unsigned int page = 0) const { return tracklets_[page].size(); }

    Tracklet *getTracklet(unsigned int i, unsigned int page = 0) { return tracklets_[page][i]; }

    unsigned int nPage() const { return npage_; }

    void clean() override;

    void writeTPAR(bool first, unsigned int iSector);

  private:
    std::vector<std::vector<Tracklet *> > tracklets_;
    unsigned int npage_;
  };

};  // namespace trklet
#endif
