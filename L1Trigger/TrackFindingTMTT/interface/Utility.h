#ifndef L1Trigger_TrackFindingTMTT_Utility_h
#define L1Trigger_TrackFindingTMTT_Utility_h

#include <vector>
#include <string>

namespace tmtt {

  class TP;
  class Stub;
  class Settings;

  namespace Utility {
    // Count number of tracker layers a given list of stubs are in.
    //
    // By default uses the "reduced" layer ID if the configuration file requested it. However,
    // you can insist on "normal" layer ID being used instead, ignoring the configuration file, by
    // std::setting disableReducedLayerID = true.
    //
    // N.B. The "reduced" layer ID merges some layer IDs, so that no more than 8 ID are needed in any
    // eta region, so as to simplify the firmware.
    //
    // N.B. You should std::set disableReducedLayerID = false when counting the number of layers on a tracking
    // particle or how many layers it shares with an L1 track. Such counts by CMS convention use "normal" layer ID.
    //
    // By default, considers both PS+2S modules, but optionally considers only the PS ones if onlyPS = true.

    enum AlgoStep { HT, SEED, DUP, FIT };

    unsigned int countLayers(const Settings* settings,
                             const std::vector<const Stub*>& stubs,
                             bool disableReducedLayerID = false,
                             bool onlyPS = false);

    unsigned int countLayers(const Settings* settings,
                             const std::vector<Stub*>& stubs,
                             bool disableReducedLayerID = false,
                             bool onlyPS = false);

    // Given a std::set of stubs (presumably on a reconstructed track candidate)
    // return the best matching Tracking Particle (if any),
    // the number of tracker layers in which one of the stubs matched one from this tracking particle,
    // and the list of the subset of the stubs which match those on the tracking particle.

    const TP* matchingTP(const Settings* settings,
                         const std::vector<const Stub*>& vstubs,
                         unsigned int& nMatchedLayersBest,
                         std::vector<const Stub*>& matchedStubsBest);

    const TP* matchingTP(const Settings* settings,
                         const std::vector<Stub*>& vstubs,
                         unsigned int& nMatchedLayersBest,
                         std::vector<const Stub*>& matchedStubsBest);

    // Determine min number of layers a track candidate must have stubs in to be defined as a track.
    // 1st argument indicates from which step in chain this function is called: HT, SEED, DUP or FIT.
    unsigned int numLayerCut(Utility::AlgoStep algo,
                             const Settings* settings,
                             unsigned int iPhiSec,
                             unsigned int iEtaReg,
                             float invPt,
                             float eta = 0.);
  }  // namespace Utility

}  // namespace tmtt

#endif
