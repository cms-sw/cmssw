#ifndef L1Trigger_TrackFindingTracklet_interface_CandidateMatchMemory_h
#define L1Trigger_TrackFindingTracklet_interface_CandidateMatchMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>
#include <string>
#include <utility>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class Tracklet;

  class CandidateMatchMemory : public MemoryBase {
  public:
    CandidateMatchMemory(std::string name, Settings const& settings);

    ~CandidateMatchMemory() override = default;

    void addMatch(std::pair<Tracklet*, int> tracklet, const Stub* stub);

    unsigned int nMatches() const { return matches_.size(); }

    std::pair<std::pair<Tracklet*, int>, const Stub*> getMatch(unsigned int i) { return matches_[i]; }

    void clean() override { matches_.clear(); }

    void writeCM(bool first, unsigned int iSector);

  private:
    std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > matches_;
  };

};  // namespace trklet
#endif
