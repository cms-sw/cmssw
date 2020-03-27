#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <vector>
#include <string>
using namespace std;

namespace TMTT {

class TP;
class Stub;
class Settings;

namespace Utility {
  // Count number of tracker layers a given list of stubs are in.
  //
  // By default uses the "reduced" layer ID if the configuration file requested it. However, 
  // you can insist on "normal" layer ID being used instead, ignoring the configuration file, by
  // setting disableReducedLayerID = true.
  //
  // N.B. The "reduced" layer ID merges some layer IDs, so that no more than 8 ID are needed in any 
  // eta region, so as to simplify the firmware.
  //
  // N.B. You should set disableReducedLayerID = false when counting the number of layers on a tracking
  // particle or how many layers it shares with an L1 track. Such counts by CMS convention use "normal" layer ID.
  //
  // By default, considers both PS+2S modules, but optionally considers only the PS ones if onlyPS = true.
  
  unsigned int countLayers(const Settings* settings, const vector<const Stub*>& stubs, bool disableReducedLayerID = false, bool onlyPS = false);

  // Given a set of stubs (presumably on a reconstructed track candidate)
  // return the best matching Tracking Particle (if any),
  // the number of tracker layers in which one of the stubs matched one from this tracking particle,
  // and the list of the subset of the stubs which match those on the tracking particle.

  const TP* matchingTP(const Settings* settings, const vector<const Stub*>& vstubs,
  	               unsigned int& nMatchedLayersBest, vector<const Stub*>& matchedStubsBest);

  // Determine the minimum number of layers a track candidate must have stubs in to be defined as a track.
  // The first argument indicates from what type of algorithm this function is called: "HT", "SEED", "DUP" or "FIT".
  unsigned int numLayerCut(string algo, const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg, float invPt, float eta = 0.);
}

}

#endif
