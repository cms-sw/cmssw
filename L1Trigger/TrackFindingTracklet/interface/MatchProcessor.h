#ifndef L1Trigger_TrackFindingTracklet_interface_MatchProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_MatchProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/ProjectionTemp.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/FullMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"

#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class Stub;
  class L1TStub;
  class Tracklet;

  class MatchProcessor : public ProcessBase {
  public:
    MatchProcessor(std::string name, Settings const& settings, Globals* global, unsigned int iSector);

    ~MatchProcessor() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    bool matchCalculator(Tracklet* tracklet, const Stub* fpgastub);

  private:
    int layer_;
    int disk_;
    bool barrel_;

    unsigned int phiregion_;

    int nvm_;      //VMs in sector
    int nvmbits_;  //# of bits for VMs in sector
    int nvmbins_;  //VMs in in phi region

    int fact_;
    int icorrshift_;
    int icorzshift_;
    int phi0shift_;

    double phioffset_;

    unsigned int phimatchcut_[N_SEED];
    unsigned int zmatchcut_[N_SEED];

    unsigned int rphicutPS_[N_SEED];
    unsigned int rphicut2S_[N_SEED];
    unsigned int rcutPS_[N_SEED];
    unsigned int rcut2S_[N_SEED];

    double phifact_;
    double rzfact_;

    int nrbits_;
    int nphiderbits_;

    AllStubsMemory* allstubs_;
    std::vector<VMStubsMEMemory*> vmstubs_;
    std::vector<TrackletProjectionsMemory*> inputprojs_;

    int ialphafactinner_[N_DSS_MOD * 2];
    int ialphafactouter_[N_DSS_MOD * 2];

    //Memory for the full matches
    std::vector<FullMatchMemory*> fullmatches_;

    //used in the layers
    std::vector<bool> table_;

    //used in the disks
    std::vector<bool> tablePS_;
    std::vector<bool> table2S_;

    unsigned int nMatchEngines_;
    std::vector<MatchEngineUnit> matchengines_;

    CircularBuffer<ProjectionTemp> inputProjBuffer_;
  };

};  // namespace trklet
#endif
