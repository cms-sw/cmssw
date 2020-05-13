#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <set>
#include <utility>
#include <cmath>
#include <iostream>

using namespace std;

namespace tmtt {

  //--- Stub window sizes copied from L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py

  const std::vector<double> DegradeBend::barrelCut_ = {0, 2, 2.5, 3.5, 4.5, 5.5, 7};
  const std::vector<std::vector<double> > DegradeBend::ringCut_ = {  // EndcapCutSet
      {0},
      {0, 1, 2.5, 2.5, 3, 2.5, 3, 3.5, 4, 4, 4.5, 3.5, 4, 4.5, 5, 5.5},
      {0, 0.5, 2.5, 2.5, 3, 2.5, 3, 3, 3.5, 3.5, 4, 3.5, 3.5, 4, 4.5, 5},
      {0, 1, 3, 3, 2.5, 3.5, 3.5, 3.5, 4, 3.5, 3.5, 4, 4.5},
      {0, 1, 2.5, 3, 2.5, 3.5, 3, 3, 3.5, 3.5, 3.5, 4, 4},
      {0, 0.5, 1.5, 3, 2.5, 3.5, 3, 3, 3.5, 4, 3.5, 4, 3.5}};
  const std::vector<std::vector<double> > DegradeBend::tiltedCut_ = {  // TiltedBarrelCutSet
      {0},
      {0, 3, 3., 2.5, 3., 3., 2.5, 2.5, 2., 1.5, 1.5, 1, 1},
      {0, 4., 4, 4, 4, 4., 4., 4.5, 5, 4., 3.5, 3.5, 3},
      {0, 5, 5, 5, 5, 5, 5, 5.5, 5, 5, 5.5, 5.5, 5.5}};
  const std::vector<double> DegradeBend::barrelNTilt_ = {0., 12., 12., 12., 0., 0., 0.};

  //--- Given the original bend, flag indicating if this is a PS or 2S module, & detector identifier,
  //--- this return the degraded stub bend, a boolean indicatng if stub bend was outside the assumed window
  //--- size programmed below, and an integer indicating how many values of the original bend
  //--- were grouped together into this single value of the degraded bend.

  void DegradeBend::degrade(float bend,
                            bool psModule,
                            const DetId& stDetId,
                            float windowFEnew,
                            float& degradedBend,
                            unsigned int& numInGroup) const {
    // Get degraded bend value.
    unsigned int windowHalfStrips;
    this->work(bend, psModule, stDetId, windowFEnew, degradedBend, numInGroup, windowHalfStrips);
  }

  //--- Does the actual work of degrading the bend.

  void DegradeBend::work(float bend,
                         bool psModule,
                         const DetId& stDetId,
                         float windowFEnew,
                         float& degradedBend,
                         unsigned int& numInGroup,
                         unsigned int& windowHalfStrips) const {
    // Calculate stub window size in half-strip units used to produce stubs.
    // This code should be kept identical to that in
    // L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc

    unsigned int window = 0;

    if (stDetId.subdetId() == StripSubdetector::TOB) {
      int layer = theTrackerTopo_->layer(stDetId);
      int ladder = theTrackerTopo_->tobRod(stDetId);
      int type = 2 * theTrackerTopo_->tobSide(stDetId) - 3;  // -1 for tilted-, 1 for tilted+, 3 for flat
      double corr = 0;

      if (type != TrackerModule::BarrelModuleType::flat)  // Only for tilted modules
      {
        corr = (barrelNTilt_.at(layer) + 1) / 2.;
        ladder =
            corr - (corr - ladder) * type;  // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
        window = 2 * (tiltedCut_.at(layer)).at(ladder);
      } else  // Classic barrel window otherwise
      {
        window = 2 * barrelCut_.at(layer);
      }
    } else if (stDetId.subdetId() == StripSubdetector::TID) {
      window = 2 * (ringCut_.at(theTrackerTopo_->tidWheel(stDetId))).at(theTrackerTopo_->tidRing(stDetId));
    }

    // Compare this with the possibly tighter window provided by the user, converting to half-strip units.
    unsigned int newWindow = (unsigned int)(2 * windowFEnew);
    if (window > newWindow)
      window = newWindow;

    // This is the window size measured in half-strips.
    windowHalfStrips = window;

    // Number of degraded bend values should correspond to 3 bits (PS modules) or 4 bits (2S modules),
    // so measuring everything in half-strip units, max integer "window" size that can be encoded without
    // compression given by 2*window+1 <= pow(2,B), where B is number of bits.
    // Hence no compression required if window cut is abs(b) < 3 (PS) or 7 (2S). Must introduce one merge for
    // each 1 unit increase in "window" beyond this.

    // Bend is measured with granularity of 0.5 strips.
    // Convert it to integer measured in half-strip units for this calculation!

    int b = std::round(2 * bend);

    if ((unsigned int)(abs(b)) <= window) {
      float degradedB;
      unsigned int numBends = 2 * window + 1;
      unsigned int numAllowed = (psModule) ? pow(2, bitsPS_) : pow(2, bits2S_);
      // Existance of bend = 0 means can only use an odd number of groups.
      numAllowed -= 1;
      if (numBends <= numAllowed) {
        // Can output uncompressed bend info.
        numInGroup = 1;
        degradedB = float(b);
      } else {
        unsigned int inSmallGroup = numBends / numAllowed;
        unsigned int numLargeGroups = numBends % numAllowed;
        unsigned int inLargeGroup = inSmallGroup + 1;
        unsigned int numSmallGroups = numAllowed - numLargeGroups;
        // Bend encoding in groups (some large, some small, one large/small, some small, some large).
        vector<unsigned int> groups;
        for (unsigned int i = 0; i < numLargeGroups / 2; i++)
          groups.push_back(inLargeGroup);
        for (unsigned int i = 0; i < numSmallGroups / 2; i++)
          groups.push_back(inSmallGroup);
        // Only one of numLargeGroups & numSmallGroups can be odd, since numAllowed is odd.
        // And whichever is odd is associated to a group with an odd number of elements since numBends is odd,
        if (numLargeGroups % 2 == 1 && inLargeGroup % 2 == 1) {
          groups.push_back(inLargeGroup);
        } else if (numSmallGroups % 2 == 1 && inSmallGroup % 2 == 1) {
          groups.push_back(inSmallGroup);
        } else {
          throw cms::Exception("LogicError") << "DegradeBend: logic error with odd numbers";
        }
        for (unsigned int i = 0; i < numSmallGroups / 2; i++)
          groups.push_back(inSmallGroup);
        for (unsigned int i = 0; i < numLargeGroups / 2; i++)
          groups.push_back(inLargeGroup);

        degradedB = 999;
        int iUp = -int(window) - 1;
        for (unsigned int& inGroup : groups) {
          iUp += inGroup;
          int iDown = iUp - inGroup + 1;
          if (b <= iUp && b >= iDown) {
            numInGroup = inGroup;
            degradedB = 0.5 * (iUp + iDown);
          }
        }
        if (degradedB == 999)
          throw cms::Exception("LogicError") << "DegradeResolution: Logic error in loop over groups";
      }

      // This is degraded bend in full strip units (neglecting bend sign).
      degradedBend = float(degradedB) / 2.;

    } else {
      // This should only happen for stubs subsequently rejected by the FE.
      numInGroup = 0;
      constexpr float rejectedStubBend = 99999.;
      degradedBend = rejectedStubBend;
    }
  }
}  // namespace tmtt
