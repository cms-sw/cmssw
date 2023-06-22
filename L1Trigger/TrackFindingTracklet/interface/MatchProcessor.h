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
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

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
    MatchProcessor(std::string name, Settings const& settings, Globals* global);

    ~MatchProcessor() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phimin);

    bool matchCalculator(Tracklet* tracklet, const Stub* fpgastub, bool print, unsigned int istep);

  private:
    unsigned int layerdisk_;
    bool barrel_;

    unsigned int phiregion_;

    int nvm_;      //VMs in sector
    int nvmbits_;  //# of bits for VMs in sector
    int nvmbins_;  //VMs in in phi region
    int nrinv_;    //# of bits for rinv

    int dzshift_;
    int icorrshift_;
    int icorzshift_;
    int phishift_;

    TrackletLUT phimatchcuttable_;
    TrackletLUT zmatchcuttable_;

    TrackletLUT rphicutPStable_;
    TrackletLUT rphicut2Stable_;
    TrackletLUT rcutPStable_;
    TrackletLUT rcut2Stable_;
    TrackletLUT alphainner_;
    TrackletLUT alphaouter_;
    TrackletLUT rSSinner_;
    TrackletLUT rSSouter_;

    TrackletLUT diskRadius_;

    int nrbits_;
    int nphiderbits_;

    //Number of r bits for the projection to use in LUT for disk
    int nrprojbits_;

    AllStubsMemory* allstubs_;
    std::vector<VMStubsMEMemory*> vmstubs_;
    std::vector<TrackletProjectionsMemory*> inputprojs_;

    int ialphafactinner_[N_DSS_MOD * 2];
    int ialphafactouter_[N_DSS_MOD * 2];

    //Memory for the full matches
    std::vector<FullMatchMemory*> fullmatches_;

    //disk projectionrinv table
    TrackletLUT rinvbendlut_;

    //LUT for bend consistency
    TrackletLUT luttable_;

    double phimin_;

    unsigned int nMatchEngines_;
    std::vector<MatchEngineUnit> matchengines_;

    int best_ideltaphi_barrel;
    int best_ideltaz_barrel;
    int best_ideltaphi_disk;
    int best_ideltar_disk;
    Tracklet* curr_tracklet;
    Tracklet* next_tracklet;

    CircularBuffer<ProjectionTemp> inputProjBuffer_;
  };

};  // namespace trklet
#endif
