#ifndef MeasurementTrackerImpl_H
#define MeasurementTrackerImpl_H

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include <map>
#include <unordered_map>
#include <vector>

class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class TkPhase2OTMeasurementDet;
class TkGluedMeasurementDet;
class TkStackMeasurementDet;
class TkDoubleSensMeasurementDet;
class GeometricSearchTracker;
class SiStripRecHitMatcher;
class GluedGeomDet;
class StackGeomDet;
class DoubleSensGeomDet;
class SiPixelFedCabling;
class TrackerGeometry;

class dso_hidden MeasurementTrackerImpl final : public MeasurementTracker {
public:
  enum QualityFlags {
    BadModules = 1,  // for everybody
    /* Strips: */ BadAPVFibers = 2,
    BadStrips = 4,
    MaskBad128StripBlocks = 8,
    /* Pixels: */ BadROCs = 2
  };

  struct BadStripCutsDet {
    StMeasurementConditionSet::BadStripCuts tib;
    StMeasurementConditionSet::BadStripCuts tob;
    StMeasurementConditionSet::BadStripCuts tid;
    StMeasurementConditionSet::BadStripCuts tec;
  };

  MeasurementTrackerImpl(const BadStripCutsDet& badStripCuts,
                         const PixelClusterParameterEstimator* pixelCPE,
                         const StripClusterParameterEstimator* stripCPE,
                         const SiStripRecHitMatcher* hitMatcher,
                         const TrackerTopology* trackerTopology,
                         const TrackerGeometry* trackerGeom,
                         const GeometricSearchTracker* geometricSearchTracker,
                         const SiStripQuality* stripQuality,
                         int stripQualityFlags,
                         int stripQualityDebugFlags,
                         const SiPixelQuality* pixelQuality,
                         const SiPixelFedCabling* pixelCabling,
                         int pixelQualityFlags,
                         int pixelQualityDebugFlags,
                         const ClusterParameterEstimator<Phase2TrackerCluster1D>* phase2OTCPE = nullptr);

  ~MeasurementTrackerImpl() override;

  const TrackerGeometry* geomTracker() const { return theTrackerGeom; }

  const GeometricSearchTracker* geometricSearchTracker() const { return theGeometricSearchTracker; }

  /// MeasurementDetSystem interface  (won't be overloaded anymore)
  MeasurementDetWithData idToDet(const DetId& id, const MeasurementTrackerEvent& data) const override {
    return MeasurementDetWithData(*idToDetBare(id, data), data);
  }

  const MeasurementDet* idToDetBare(const DetId& id, const MeasurementTrackerEvent& data) const { return findDet(id); }

  const MeasurementDet* findDet(const DetId& id) const {
    auto it = theDetMap.find(id);
    if (it != theDetMap.end()) {
      return it->second;
    } else {
      //throw exception;
    }

    return nullptr;  //to avoid compile warning
  }

  typedef std::unordered_map<unsigned int, MeasurementDet*> DetContainer;

  /// For debug only
  const DetContainer& allDets() const { return theDetMap; }
  const std::vector<TkStripMeasurementDet>& stripDets() const { return theStripDets; }
  const std::vector<TkPixelMeasurementDet>& pixelDets() const { return thePixelDets; }
  const std::vector<TkGluedMeasurementDet>& gluedDets() const { return theGluedDets; }
  const std::vector<TkStackMeasurementDet>& stackDets() const { return theStackDets; }
  const std::vector<TkDoubleSensMeasurementDet>& doubleSensGeomDets() const { return theDoubleSensGeomDets; }

  const StMeasurementConditionSet& stripDetConditions() const override { return theStDetConditions; }
  const PxMeasurementConditionSet& pixelDetConditions() const override { return thePxDetConditions; }
  const Phase2OTMeasurementConditionSet& phase2DetConditions() const override { return thePhase2DetConditions; }

protected:
  StMeasurementConditionSet theStDetConditions;
  PxMeasurementConditionSet thePxDetConditions;
  Phase2OTMeasurementConditionSet thePhase2DetConditions;

  DetContainer theDetMap;

  std::vector<TkPixelMeasurementDet> thePixelDets;
  std::vector<TkStripMeasurementDet> theStripDets;
  std::vector<TkPhase2OTMeasurementDet> thePhase2Dets;
  std::vector<TkGluedMeasurementDet> theGluedDets;
  std::vector<TkStackMeasurementDet> theStackDets;
  std::vector<TkDoubleSensMeasurementDet> theDoubleSensGeomDets;

  const SiPixelFedCabling* thePixelCabling;

  void initialize(const TrackerTopology* trackerTopology);
  void initStMeasurementConditionSet(std::vector<TkStripMeasurementDet>& stripDets);
  void initPxMeasurementConditionSet(std::vector<TkPixelMeasurementDet>& pixelDets);
  void initPhase2OTMeasurementConditionSet(std::vector<TkPhase2OTMeasurementDet>& phase2Dets);

  void addStripDet(const GeomDet* gd);
  void addPixelDet(const GeomDet* gd);
  void addPhase2Det(const GeomDet* gd);

  void addGluedDet(const GluedGeomDet* gd);
  void addStackDet(const StackGeomDet* gd);
  void addDoubleSensGeomDet(const DoubleSensGeomDet* gd);

  void initGluedDet(TkGluedMeasurementDet& det, const TrackerTopology* trackerTopology);
  void initStackDet(TkStackMeasurementDet& det);
  void initDoubleSensDet(TkDoubleSensMeasurementDet& det);

  void addDets(const TrackingGeometry::DetContainer& dets, bool subIsPixel, bool subIsOT);

  bool checkDets();

  void initializeStripStatus(const BadStripCutsDet& badStripCuts,
                             const SiStripQuality* stripQuality,
                             int qualityFlags,
                             int qualityDebugFlags);

  void initializePixelStatus(const SiPixelQuality* stripQuality,
                             const SiPixelFedCabling* pixelCabling,
                             int qualityFlags,
                             int qualityDebugFlags);
};

#endif
