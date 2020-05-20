#ifndef L1Trigger_TrackFindingTracklet_interface_MatchCalculator_h
#define L1Trigger_TrackFindingTracklet_interface_MatchCalculator_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

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
    MatchCalculator(std::string name, const Settings* settings, Globals* global, unsigned int iSector);

    ~MatchCalculator() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > mergeMatches(
        std::vector<CandidateMatchMemory*>& candmatch);

  private:
    unsigned int layerdisk_;
    unsigned int phiregion_;

    int fact_;
    int icorrshift_;
    int icorzshift_;
    int phi0shift_;
    double phioffset_;

    unsigned int phimatchcut_[N_SEEDINDEX];
    unsigned int zmatchcut_[N_SEEDINDEX];
    unsigned int rphicutPS_[N_SEEDINDEX];
    unsigned int rphicut2S_[N_SEEDINDEX];
    unsigned int rcutPS_[N_SEEDINDEX];
    unsigned int rcut2S_[N_SEEDINDEX];

    int ialphafactinner_[10];
    int ialphafactouter_[10];

    AllStubsMemory* allstubs_;
    AllProjectionsMemory* allprojs_;

    std::vector<CandidateMatchMemory*> matches_;
    std::vector<FullMatchMemory*> fullMatches_;
  };

};  // namespace trklet
#endif
