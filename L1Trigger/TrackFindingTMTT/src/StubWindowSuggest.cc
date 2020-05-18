#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <mutex>

using namespace std;

namespace {
  std::mutex myMutex;
}

namespace tmtt {

  //=== Analyse stub window required for this stub.

  void StubWindowSuggest::process(const TrackerTopology* trackerTopo, const Stub* stub) {
    std::lock_guard<std::mutex> myGuard(myMutex);  // Allow only one thread.

    // Half-size of FE chip bend window corresponding to Pt range in which tracks are to be found.
    const double invPtMax = 1 / ptMin_;
    double bendHalfWind = invPtMax / std::abs(stub->qOverPtOverBend());
    // Increase half-indow size to allow for resolution in bend.
    bendHalfWind += stub->bendCutInFrontend();
    // Stub bend is measured here in half-integer values.
    bendHalfWind = int(2 * bendHalfWind) / 2.;

    // Compare with half-size of FE bend window stored in arrays.
    this->updateStoredWindow(trackerTopo, stub, bendHalfWind);
  }

  //===  Update stored stub window size with this stub.

  void StubWindowSuggest::updateStoredWindow(const TrackerTopology* trackerTopo,
                                             const Stub* stub,
                                             double bendHalfWind) {
    // Code accessing geometry inspired by L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc

    DetId stDetId(stub->trackerModule()->detId());

    double* storedHalfWindow = sw_.storedWindowSize(trackerTopo, stDetId);
    /*
    if (stDetId.subdetId() == StripSubdetector::TOB) {
      unsigned int layer = trackerTopo->layer(stDetId);
      unsigned int ladder = trackerTopo->tobRod(stDetId);
      int type = 2 * trackerTopo->tobSide(stDetId) - 3;  // -1 for tilted-, 1 for tilted+, 3 for flat
      double corr = 0;

      if (type != TrackerModule::BarrelModuleType::flat)  // Only for tilted modules
      {
        corr = (sw_.numTiltedLayerRings().at(layer) + 1) / 2.;
	// Corrected ring number, between 0 and barrelNTilt.at(layer), in ascending |z|
        ladder = corr - (corr - ladder) * type;  
        double& storedHalfWindow = (sw_.windowSizeTiltedLayersRings().at(layer)).at(ladder);
        if (storedHalfWindow < bendHalfWind)
          storedHalfWindow = bendHalfWind;
      } else  // Classic barrel window otherwise
      {
        double& storedHalfWindow = sw_.windowSizeBarrelLayers().at(layer);
        if (storedHalfWindow < bendHalfWind)
          storedHalfWindow = bendHalfWind;
      }

    } else if (stDetId.subdetId() == StripSubdetector::TID) {
      unsigned int wheel = trackerTopo->tidWheel(stDetId);
      unsigned int ring = trackerTopo->tidRing(stDetId);
      double& storedHalfWindow = sw_.windowSizeEndcapDisksRings().at(wheel).at(ring);
      if (storedHalfWindow < bendHalfWind)
        storedHalfWindow = bendHalfWind;
    }
*/
    if (*storedHalfWindow < bendHalfWind)
      *storedHalfWindow = bendHalfWind;
  }

  //=== Print results (should be done in endJob();

  void StubWindowSuggest::printResults() const {
    PrintL1trk(1) << "==============================================================================";
    PrintL1trk(1) << " Stub window sizes that TMTT suggests putting inside ";
    PrintL1trk(1) << "   /L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py";
    PrintL1trk(1) << " (These should give good efficiency, but tighter windows may be needed to";
    PrintL1trk(1) << "  limit the data rate from the FE tracker electronics).";
    PrintL1trk(1) << "==============================================================================";

    std::stringstream text;
    string div;

    text << "BarrelCut = cms.vdouble( ";
    div = "";
    for (const auto& cut : sw_.windowSizeBarrelLayers()) {
      text << div << cut;
      div = ", ";
    }
    text << "),";
    PrintL1trk(1) << text.str();

    PrintL1trk(1) << "TiltedBarrelCutSet = cms.VPSET( ";
    for (const auto& cutVec : sw_.windowSizeTiltedLayersRings()) {
      text.str("");
      text << "     cms.PSet( TiltedCut = cms.vdouble(";
      if (cutVec.empty())
        text << "0";
      div = "";
      for (const auto& cut : cutVec) {
        text << div << cut;
        div = ", ";
      }
      text << "), ),";
      PrintL1trk(1) << text.str();
    }
    PrintL1trk(1) << "),";

    PrintL1trk(1) << "EndcapCutSet = cms.VPSET( ";
    for (const auto& cutVec : sw_.windowSizeEndcapDisksRings()) {
      text.str("");
      text << "     cms.PSet( EndcapCut = cms.vdouble(";
      if (cutVec.empty())
        text << "0";
      div = "";
      for (const auto& cut : cutVec) {
        text << div << cut;
        div = ", ";
      }
      text << "), ),";
      PrintL1trk(1) << text.str();
    }
    PrintL1trk(1) << ")";

    PrintL1trk(1) << "==============================================================================";
  }

}  // namespace tmtt
