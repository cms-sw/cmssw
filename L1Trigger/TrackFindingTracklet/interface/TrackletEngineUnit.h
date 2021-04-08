#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletEngineUnit_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletEngineUnit_h

#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"

#include <cassert>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class FPGAWord;

  struct TEData {
    const Stub* stub_;
    int start_;
    int rzbinfirst_;
    int rzdiffmax_;
    int innerfinephi_;
    FPGAWord innerbend_;
    std::vector<std::tuple<int, int, int> > regions_;  // next z/r bin; phi-region; nstub
  };

  class TrackletEngineUnit {
  public:
    TrackletEngineUnit(const Settings* const settings,
                       unsigned int nbitsfinephi,
                       unsigned int layerdisk2,
                       unsigned int iSeed,
                       unsigned int nbitsfinephiediff,
                       unsigned int iAllStub,
                       std::vector<bool> const& pttableinner,
                       std::vector<bool> const& pttableouter,
                       VMStubsTEMemory* outervmstubs);

    ~TrackletEngineUnit() = default;

    void init(const TEData& tedata);

    bool empty() const { return candpairs_.empty(); }

    const std::pair<const Stub*, const Stub*>& read() { return candpairs_.read(); }

    const std::pair<const Stub*, const Stub*>& peek() const { return candpairs_.peek(); }

    bool idle() const { return idle_; }

    void reset();

    void step();

  private:
    VMStubsTEMemory* outervmstubs_;
    TEData tedata_;
    const Settings* settings_;
    unsigned int nbitsfinephi_;
    unsigned int layerdisk2_;
    unsigned int iSeed_;
    unsigned int nbitsfinephidiff_;

    unsigned int iAllStub_;

    //unsigned int memory slot
    unsigned int nreg_;
    unsigned int istub_;
    unsigned int ireg_;
    unsigned int next_;
    unsigned int nstub_;

    bool idle_;

    std::vector<bool> pttableinner_;
    std::vector<bool> pttableouter_;

    //save the candidate matches
    CircularBuffer<std::pair<const Stub*, const Stub*> > candpairs_;
  };  // TrackletEngineUnit

};  // namespace trklet
#endif
