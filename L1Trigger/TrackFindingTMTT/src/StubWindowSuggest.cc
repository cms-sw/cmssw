#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <mutex>

using namespace std;

namespace tmtt {

  //=== Get FE window size arrays (via copy) used with stub producer, but set to zero.

  void StubWindowSuggest::setFEWindows(const StubFEWindows* sw) {
    static std::mutex myMutex;
    std::lock_guard<std::mutex> myGuard(myMutex);  // Allow only one thread.
    // Only need to create FE windows once.
    if (not sw_) {
      sw_ = std::make_unique<StubFEWindows>(*sw);  // Copy
      sw_->setZero();
    }
  }

  //=== Analyse stub window required for this stub.

  void StubWindowSuggest::process(const TrackerTopology* trackerTopo, const Stub* stub) {
    static std::mutex myMutex;
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

    double* storedHalfWindow = sw_->storedWindowSize(trackerTopo, stDetId);

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
    for (const auto& cut : sw_->windowSizeBarrelLayers()) {
      text << div << cut;
      div = ", ";
    }
    text << "),";
    PrintL1trk(1) << text.str();

    PrintL1trk(1) << "TiltedBarrelCutSet = cms.VPSET( ";
    for (const auto& cutVec : sw_->windowSizeTiltedLayersRings()) {
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
    for (const auto& cutVec : sw_->windowSizeEndcapDisksRings()) {
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
