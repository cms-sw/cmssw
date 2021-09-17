#ifndef L1Trigger_TrackFindingTMTT_DegradeBend_h
#define L1Trigger_TrackFindingTMTT_DegradeBend_h

#include "L1Trigger/TrackFindingTMTT/interface/StubFEWindows.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

class TrackerTopology;

namespace tmtt {

  class DegradeBend {
    /*
   *-------------------------------------------------------------------------------------------------------------------
   * Implements reduced bits to encode stub bend information: 3 bits for PS, 4 bits for 2S, since the Tracker
   * doesn't have the bandwidth to output the unreduced data from the FE electronics.
   *
   * This obtains the stub window sizes from L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py ,
   * which must be loaded into the cfg file (with the same params used originally to make the stubs).
   * 
   * The TMTT L1 tracking code can optionally tighten these windows further (cfg option "KillLowPtStubs").  
   * This gives slightly more granular encoding with Pt > 3 GeV.
   * 
   * TMTT histograms "hisBendFEVsLayerOrRingPS" & "hisBendFEVsLayerOrRing2S" produced by the "Histos" class
   * are useful for debugging.   *-------------------------------------------------------------------------------------------------------------------
   */

  public:
    typedef TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> StubAlgorithmOfficial;

    DegradeBend(const TrackerTopology* trackerTopo, const StubFEWindows* sw, const StubAlgorithmOfficial* stubAlgo)
        : theTrackerTopo_(trackerTopo), sw_(sw), stubAlgo_(stubAlgo) {}

    // Given the original bend, flag indicating if this is a PS or 2S module, & detector identifier,
    // this return the degraded stub bend, a boolean indicatng if stub bend was outside the assumed window
    // size programmed below, and an integer indicating how many values of the original bend
    // were grouped together into this single value of the degraded bend.
    //
    // (Input argument windowFEnew specifies the stub window size that should be used for this stub instead
    // of the window sizes specified in TTStubAlgorithmRegister_cfi.py , but it will ONLY replace the latter
    // sizes if it windowFEnew is smaller. If you always want to use TTStubAlgorithmRegister_cfi.py, then
    // std::set windowFEnew to a large number, such as 99999.).
    void degrade(float bend,
                 bool psModule,
                 const DetId& stDetId,
                 float windowFEnew,
                 float& degradedBend,
                 unsigned int& numInGroup) const;

  private:
    // Does the actual work of degrading the bend.
    void work(float bend,
              bool psModule,
              const DetId& stDetId,
              float windowFEnew,
              float& degradedBend,
              unsigned int& numInGroup,
              unsigned int& windowHalfStrips) const;

  private:
    const TrackerTopology* theTrackerTopo_;

    // Stub window sizes as encoded in L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h
    const StubFEWindows* sw_;

    // TTStub produce algo used to make stubs.
    const StubAlgorithmOfficial* stubAlgo_;
  };

}  // namespace tmtt
#endif
