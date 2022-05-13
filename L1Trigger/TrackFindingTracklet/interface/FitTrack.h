#ifndef L1Trigger_TrackFindingTracklet_interface_FitTrack_H
#define L1Trigger_TrackFindingTracklet_interface_FitTrack_H

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/FullMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackFitMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubStreamData.h"

#include <vector>
#include <deque>

namespace trklet {

  class Settings;
  class Globals;
  class Stub;

  class FitTrack : public ProcessBase {
  public:
    FitTrack(std::string name, Settings const& settings, Globals* global);

    ~FitTrack() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;

    void addInput(MemoryBase* memory, std::string input) override;

    // used if USEHYBRID is not defined
    void trackFitChisq(Tracklet* tracklet, std::vector<const Stub*>&, std::vector<std::pair<int, int>>&);

    // used if USEHYBRID is defined
    void trackFitKF(Tracklet* tracklet,
                    std::vector<const Stub*>& trackstublist,
                    std::vector<std::pair<int, int>>& stubidslist);

    // used for propagating tracklet without fitting
    void trackFitFake(Tracklet* tracklet, std::vector<const Stub*>&, std::vector<std::pair<int, int>>&);

    std::vector<Tracklet*> orderedMatches(std::vector<FullMatchMemory*>& fullmatch);

    void execute(std::deque<std::string>& streamTrackRaw,
                 std::vector<std::deque<StubStreamData>>& stubStream,
                 unsigned int iSector);

  private:
    std::vector<TrackletParametersMemory*> seedtracklet_;
    std::vector<FullMatchMemory*> fullmatch1_;
    std::vector<FullMatchMemory*> fullmatch2_;
    std::vector<FullMatchMemory*> fullmatch3_;
    std::vector<FullMatchMemory*> fullmatch4_;

    unsigned int iSector_;

    TrackFitMemory* trackfit_;
  };

};  // namespace trklet
#endif
