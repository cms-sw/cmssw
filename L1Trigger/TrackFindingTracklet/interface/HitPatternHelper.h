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
#include "L1Trigger/TrackFindingTracklet/interface/HitPatternHelperRcd.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <bitset>
#include <iostream>
#include <vector>
#include <utility>
#include <map>

namespace hph {

  //Class that stores configuration for HitPatternHelper
  class Setup {
  public:
    Setup(const edm::ParameterSet& iConfig,
          const tt::Setup& setupTT,
          const trackerTFP::DataFormats& dataFormats,
          const trackerTFP::LayerEncoding& layerEncoding);
    ~Setup() {}

    bool hphDebug() const { return hphDebug_; }
    bool useNewKF() const { return useNewKF_; }
    double chosenRofZ() const { return chosenRofZ_; }
    std::vector<double> etaRegions() const { return etaRegions_; }
    std::map<int, std::map<int, std::vector<int>>> layermap() const { return layermap_; }
    int nKalmanLayers() const { return nKalmanLayers_; }
    int etaRegion(double z0, double cot, bool useNewKF) const;
    int digiCot(double cot, int binEta) const;
    int digiZT(double z0, double cot, int binEta) const;
    const std::vector<int>& layerEncoding(int binEta, int binZT, int binCot) const {
      return layerEncoding_.layerEncoding(binEta, binZT, binCot);
    }
    const std::map<int, const tt::SensorModule*>& layerEncodingMap(int binEta, int binZT, int binCot) const {
      return layerEncoding_.layerEncodingMap(binEta, binZT, binCot);
    }

  private:
    edm::ParameterSet iConfig_;
    edm::ParameterSet oldKFPSet_;
    const tt::Setup setupTT_;  // Helper class to store TrackTrigger configuration
    const trackerTFP::DataFormats dataFormats_;
    const trackerTFP::DataFormat dfcot_;
    const trackerTFP::DataFormat dfzT_;
    const trackerTFP::LayerEncoding layerEncoding_;
    bool hphDebug_;
    bool useNewKF_;
    double chosenRofZNewKF_;
    std::vector<double> etaRegionsNewKF_;
    double chosenRofZ_;
    std::vector<double> etaRegions_;
    std::map<int, std::map<int, std::vector<int>>> layermap_;  // Hard-coded layermap in Old KF
    int nEtaRegions_;                                          // # of eta regions
    int nKalmanLayers_;                                        // # of maximum KF layers allowed
  };                                                           // Only needed by Old KF

  //Class that returns decoded information from hitpattern
  class HitPatternHelper {
  public:
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
    std::vector<float> bonusFeatures() { return bonusFeatures_; }  //bonus features for track quality

    int reducedId(
        int layerId);  //Converts layer ID (1~6->L1~L6;11~15->D1~D5) to reduced layer ID (0~5->L1~L6;6~10->D1~D5)
    int findLayer(int layerId);  //Search for a layer ID from sensor modules

  private:
    const Setup* setup_;
    bool hphDebug_;
    bool useNewKF_;
    std::vector<double> etaRegions_;
    std::map<int, std::map<int, std::vector<int>>> layermap_;
    int nKalmanLayers_;
    int etaBin_;
    int cotBin_;
    int zTBin_;
    std::vector<int> layerEncoding_;
    std::map<int, const tt::SensorModule*> layerEncodingMap_;
    int numExpLayer_;
    int hitpattern_;
    int etaSector_;
    int numMissingLayer_;
    int numMissingPS_;
    int numMissing2S_;
    int numPS_;
    int num2S_;
    int numMissingInterior1_;
    int numMissingInterior2_;
    std::vector<int> binary_;
    std::vector<float> bonusFeatures_;
  };

}  // namespace hph

EVENTSETUP_DATA_DEFAULT_RECORD(hph::Setup, hph::SetupRcd);

#endif
