#ifndef L1Trigger_TrackFindingTracklet_interface_MatchCalculator_h
#define L1Trigger_TrackFindingTracklet_interface_MatchCalculator_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

#include <string>
#include <vector>

namespace trklet {

  class Globals;
  class Stub;
  class L1TStub;
  class Tracklet;
  class AllStubsMemory;
  class AllProjectionsMemory;
  class CandidateMatchMemory;
  class FullMatchMemory;

  class MatchCalculator : public ProcessBase {
  public:
    MatchCalculator(std::string name, Settings const& settings, Globals* global);

    ~MatchCalculator() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phioffset);

    std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > mergeMatches(
        std::vector<CandidateMatchMemory*>& candmatch);

  private:
    unsigned int layerdisk_;
    unsigned int phiregion_;

    int fact_;
    int icorrshift_;
    int icorzshift_;
    int phi0shift_;

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

    int ialphafactinner_[N_DSS_MOD * 2];
    int ialphafactouter_[N_DSS_MOD * 2];

    AllStubsMemory* allstubs_;
    AllProjectionsMemory* allprojs_;

    std::vector<CandidateMatchMemory*> matches_;
    std::vector<FullMatchMemory*> fullMatches_;
  };

};  // namespace trklet
#endif
