#ifndef L1Trigger_TrackFindingTracklet_interface_MatchEngine_h
#define L1Trigger_TrackFindingTracklet_interface_MatchEngine_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class VMStubsMEMemory;
  class VMProjectionsMemory;
  class CandidateMatchMemory;

  class MatchEngine : public ProcessBase {
  public:
    MatchEngine(std::string name, Settings const& settings, Globals* global);

    ~MatchEngine() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

  private:
    VMStubsMEMemory* vmstubs_;
    VMProjectionsMemory* vmprojs_;

    CandidateMatchMemory* candmatches_;

    unsigned int layerdisk_;

    bool barrel_;
    int nvm_;
    int nvmbits_;

    unsigned int nfinephibits_;  //number of fine phi bits in phi VM bin
    unsigned int nrinv_;         //number of bits for rinv in stub bend LUT

    //used in the layers
    std::vector<bool> table_;

    //used in the disks
    std::vector<bool> tablePS_;
    std::vector<bool> table2S_;
  };

};  // namespace trklet
#endif
