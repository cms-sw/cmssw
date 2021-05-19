
#include "L1Trigger/VertexFinder/interface/utility.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"
#include "L1Trigger/VertexFinder/interface/Stub.h"
#include "L1Trigger/VertexFinder/interface/TP.h"

using namespace std;

namespace l1tVertexFinder {
  namespace utility {

    //=== Count number of tracker layers a given list of stubs are in.
    //=== By default, consider both PS+2S modules, but optionally consider only the PS ones.

    unsigned int countLayers(const AnalysisSettings& settings, const vector<const Stub*>& vstubs, bool onlyPS) {
      //=== Unpack configuration parameters

      // Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
      static bool useLayerID = settings.useLayerID();
      // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
      static float layerIDfromRadiusBin = settings.layerIDfromRadiusBin();
      // Inner radius of tracker.
      static float trackerInnerRadius = settings.trackerInnerRadius();

      const int maxLayerID(30);
      vector<bool> foundLayers(maxLayerID, false);

      if (useLayerID) {
        // Count layers using CMSSW layer ID.
        for (const Stub* stub : vstubs) {
          if ((!onlyPS) || stub->psModule()) {  // Consider only stubs in PS modules if that option specified.
            int layerID = stub->layerId();
            if (layerID >= 0 && layerID < maxLayerID) {
              foundLayers[layerID] = true;
            } else {
              throw cms::Exception("Utility::invalid layer ID");
            }
          }
        }
      } else {
        // Count layers by binning stub distance from beam line.
        for (const Stub* stub : vstubs) {
          if ((!onlyPS) || stub->psModule()) {  // Consider only stubs in PS modules if that option specified.
            int layerID = (int)((stub->r() - trackerInnerRadius) / layerIDfromRadiusBin);
            if (layerID >= 0 && layerID < maxLayerID) {
              foundLayers[layerID] = true;
            } else {
              throw cms::Exception("Utility::invalid layer ID");
            }
          }
        }
      }

      unsigned int ncount = 0;
      for (const bool& found : foundLayers) {
        if (found)
          ncount++;
      }

      return ncount;
    }
  }  // end namespace utility
}  // end namespace l1tVertexFinder
