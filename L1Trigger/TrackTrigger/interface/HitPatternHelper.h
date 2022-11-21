// This is a helper function that can be used to decode hitpattern, which is a 7-bit integer produced by the Kalman filter (KF).
// Hitpattern is stored at TTTrack objects (DataFormats/L1TrackTrigger/interface/TTTrack.h)
// It can be accessed via TTTrack: hitPattern()
//
// There are two classes declared in HitPatternHelper (hph) namesapce:
// 1)Setup: This is used to produce a layermap and a collection of <tt::SensorModule> needed by HitPatternHelper.
// 2)HitPatternHelper: This function returns more specific information (e.g. module type, layer id,...) about each stub on the TTTrack objects.
// This function needs three variables from TTTrack: hitPattern(),tanL() and z0().
// It makes predictions in two different ways depending on which version of the KF is deployed:
//
// Old KF (L1Trigger/TrackFindingTMTT/plugins/TMTrackProducer.cc) is a CMSSW emulation of KF firmware that
// ignores truncation effect and is not bit/clock cycle accurate.
// With this version of the KF, HitPatternHelper relys on a hard-coded layermap to deterimne layer IDs of each stub.
//
// New KF (L1Trigger/TrackerTFP/plugins/ProducerKF.cc) is a new CMSSW emulation of KF firmware that
// considers truncaton effect and is bit/clock cycle accurate.
// With this version of the KF, HitPatternHelper makes predictions based on the spatial coordinates of tracks and detector modules.
//
//  Created by J.Li on 1/23/21.
//

#ifndef L1Trigger_TrackTrigger_interface_HitPatternHelper_h
#define L1Trigger_TrackTrigger_interface_HitPatternHelper_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "L1Trigger/TrackTrigger/interface/HitPatternHelperRcd.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <bitset>
#include <iostream>
#include <vector>
#include <utility>
#include <map>

namespace hph {

  //Class that stores configuration for HitPatternHelper
  class Setup {
  public:
    Setup() {}
    Setup(const edm::ParameterSet& iConfig, const tt::Setup& setupTT);
    ~Setup() {}

    bool hphDebug() const { return iConfig_.getParameter<bool>("hphDebug"); }
    bool useNewKF() const { return iConfig_.getParameter<bool>("useNewKF"); }
    double deltaTanL() const { return iConfig_.getParameter<double>("deltaTanL"); }
    double chosenRofZ() const { return setupTT_.chosenRofZ(); }
    std::vector<double> etaRegions() const { return setupTT_.boundarieEta(); }
    std::vector<tt::SensorModule> sensorModules() const { return setupTT_.sensorModules(); }
    std::map<int, std::map<int, std::vector<int>>> layermap() const { return layermap_; }
    int nKalmanLayers() const { return nKalmanLayers_; }
    static auto smallerID(std::pair<int, bool> lhs, std::pair<int, bool> rhs) { return lhs.first < rhs.first; }
    static auto equalID(std::pair<int, bool> lhs, std::pair<int, bool> rhs) { return lhs.first == rhs.first; }

  private:
    edm::ParameterSet iConfig_;
    const tt::Setup setupTT_;  // Helper class to store TrackTrigger configuration
    std::vector<std::pair<int, bool>>
        layerIds_;  // layer IDs (1~6->L1~L6;11~15->D1~D5) and whether or not they are from tracker barrel
                    // Only needed by Old KF
    std::map<int, std::map<int, std::vector<int>>> layermap_;  // Hard-coded layermap in Old KF
    int nEtaRegions_;                                          // # of eta regions
    int nKalmanLayers_;                                        // # of maximum KF layers allowed
  };                                                           // Only needed by Old KF

  //Class that returns decoded information from hitpattern
  class HitPatternHelper {
  public:
    HitPatternHelper() {}
    HitPatternHelper(const Setup* setup, int hitpattern, double cot, double z0);
    ~HitPatternHelper() {}

    int etaSector() { return etaSector_; }      //Eta sectors defined in KF
    int numExpLayer() { return numExpLayer_; }  //The number of layers KF expects
    int numMissingPS() {
      return numMissingPS_;
    }  //The number of PS layers that are missing. It includes layers that are missing:
       //1)before the innermost stub on the track,
       //2)after the outermost stub on the track.
    int numMissing2S() {
      return numMissing2S_;
    }  //The number of 2S layers that are missing. It includes the two types of layers mentioned above.
    int numPS() { return numPS_; }  //The number of PS layers are found in hitpattern
    int num2S() { return num2S_; }  //The number of 2S layers are found in hitpattern
    int numMissingInterior1() {
      return numMissingInterior1_;
    }  //The number of missing interior layers (using only hitpattern)
    int numMissingInterior2() {
      return numMissingInterior2_;
    }  //The number of missing interior layers (using hitpattern, layermap from Old KF and sensor modules)
    std::vector<int> binary() { return binary_; }  //11-bit hitmask needed by TrackQuality.cc (0~5->L1~L6;6~10->D1~D5)
    static auto smallerID(tt::SensorModule lhs, tt::SensorModule rhs) { return lhs.layerId() < rhs.layerId(); }
    static auto equalID(tt::SensorModule lhs, tt::SensorModule rhs) { return lhs.layerId() == rhs.layerId(); }

    int ReducedId(
        int layerId);  //Converts layer ID (1~6->L1~L6;11~15->D1~D5) to reduced layer ID (0~5->L1~L6;6~10->D1~D5)
    int findLayer(int layerId);  //Search for a layer ID from sensor modules

  private:
    int etaSector_;
    int hitpattern_;
    int numExpLayer_;
    int numMissingLayer_;
    int numMissingPS_;
    int numMissing2S_;
    int numPS_;
    int num2S_;
    int numMissingInterior1_;
    int numMissingInterior2_;
    double cot_;
    double z0_;
    const Setup* setup_;
    std::vector<tt::SensorModule> layers_;  //Sensor modules that particles are expected to hit
    std::vector<int> binary_;
    bool hphDebug_;
    bool useNewKF_;
    float chosenRofZ_;
    float deltaTanL_;  // Uncertainty added to tanL (cot) when layermap in new KF is determined
    std::vector<double> etaRegions_;
    int nKalmanLayers_;
    std::map<int, std::map<int, std::vector<int>>> layermap_;
  };

}  // namespace hph

EVENTSETUP_DATA_DEFAULT_RECORD(hph::Setup, hph::SetupRcd);

#endif
