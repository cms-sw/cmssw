#ifndef L1Trigger_TrackFindingTMTT_MiniHTstage_h
#define L1Trigger_TrackFindingTMTT_MiniHTstage_h

#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/MuxHToutputs.h"
#include "L1Trigger/TrackFindingTMTT/interface/Array2D.h"

#include <memory>

namespace tmtt {

  class Settings;

  class MiniHTstage {
  public:
    MiniHTstage(const Settings* settings);

    void exec(Array2D<std::unique_ptr<HTrphi>>& mHtRphis);

  private:
    // Do load balancing
    unsigned int linkIDLoadBalanced(
        unsigned int link,
        unsigned int mBin,
        unsigned int cBin,
        unsigned int numStubs,
        std::map<std::pair<unsigned int, unsigned int>, unsigned int>& numStubsPerLinkStage1,
        std::map<std::pair<unsigned int, unsigned int>, unsigned int>& numStubsPerLinkStage2,
        bool test = false) const;

  private:
    const Settings* settings_;  // Configuration parameters
    bool miniHTstage_;
    MuxHToutputs::MuxAlgoName muxOutputsHT_;
    unsigned int houghNbinsPt_;
    unsigned int houghNbinsPhi_;
    unsigned int miniHoughLoadBalance_;
    unsigned int miniHoughNbinsPt_;
    unsigned int miniHoughNbinsPhi_;
    float miniHoughMinPt_;
    bool miniHoughDontKill_;
    float miniHoughDontKillMinPt_;
    unsigned int numSubSecsEta_;
    unsigned int numPhiNonants_;
    unsigned int numPhiSecPerNon_;
    unsigned int numEtaRegions_;
    bool busySectorKill_;
    unsigned int busySectorNumStubs_;
    std::vector<unsigned int> busySectorMbinRanges_;
    float chosenRofPhi_;
    float binSizeQoverPtAxis_;
    float binSizePhiTrkAxis_;
    float invPtToDphi_;
    unsigned int nMiniHTcells_;
    unsigned int nHTlinksPerNonant_;
  };

}  // namespace tmtt

#endif
