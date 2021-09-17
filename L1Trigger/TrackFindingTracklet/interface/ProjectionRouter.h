#ifndef L1Trigger_TrackFindingTracklet_interface_ProjectionRouter_h
#define L1Trigger_TrackFindingTracklet_interface_ProjectionRouter_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMProjectionsMemory.h"

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;

  class ProjectionRouter : public ProcessBase {
  public:
    ProjectionRouter(std::string name, Settings const& settings, Globals* global);

    ~ProjectionRouter() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

  private:
    unsigned int layerdisk_;

    int nrbits_;
    int nphiderbits_;

    //disk projectionrinv table
    TrackletLUT rinvbendlut_;

    std::vector<TrackletProjectionsMemory*> inputproj_;

    AllProjectionsMemory* allproj_;
    std::vector<VMProjectionsMemory*> vmprojs_;
  };

};  // namespace trklet
#endif
