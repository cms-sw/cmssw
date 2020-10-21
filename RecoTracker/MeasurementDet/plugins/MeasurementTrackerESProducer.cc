#include "MeasurementTrackerImpl.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include <string>
#include <memory>

class dso_hidden MeasurementTrackerESProducer : public edm::ESProducer {
public:
  MeasurementTrackerESProducer(const edm::ParameterSet &p);
  ~MeasurementTrackerESProducer() override;
  std::unique_ptr<MeasurementTracker> produce(const CkfComponentsRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> pixelQualityToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> pixelCablingToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;

  edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEToken_;
  edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> stripCPEToken_;
  edm::ESGetToken<SiStripRecHitMatcher, TkStripCPERecord> hitMatcherToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> geometricSearchTrackerToken_;
  edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> phase2TrackerCPEToken_;

  MeasurementTrackerImpl::BadStripCutsDet badStripCuts_;

  int pixelQualityFlags_;
  int pixelQualityDebugFlags_;

  int stripQualityFlags_;
  int stripQualityDebugFlags_;

  bool usePhase2_ = false;
};

using namespace edm;

namespace {
  std::pair<int, int> stripFlags(const edm::ParameterSet &pset) {
    int stripQualityFlags = 0;
    int stripQualityDebugFlags = 0;

    if (pset.getParameter<bool>("UseStripModuleQualityDB")) {
      stripQualityFlags += MeasurementTracker::BadModules;
      if (pset.getUntrackedParameter<bool>("DebugStripModuleQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadModules;
      }
    }
    if (pset.getParameter<bool>("UseStripAPVFiberQualityDB")) {
      stripQualityFlags += MeasurementTracker::BadAPVFibers;
      if (pset.getUntrackedParameter<bool>("DebugStripAPVFiberQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadAPVFibers;
      }
      if (pset.getParameter<bool>("MaskBadAPVFibers")) {
        stripQualityFlags += MeasurementTracker::MaskBad128StripBlocks;
      }
    }
    if (pset.getParameter<bool>("UseStripStripQualityDB")) {
      stripQualityFlags += MeasurementTracker::BadStrips;
      if (pset.getUntrackedParameter<bool>("DebugStripStripQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadStrips;
      }
    }
    return std::make_pair(stripQualityFlags, stripQualityDebugFlags);
  }

  std::pair<int, int> pixelFlags(const edm::ParameterSet &pset) {
    int pixelQualityFlags = 0;
    int pixelQualityDebugFlags = 0;

    if (pset.getParameter<bool>("UsePixelModuleQualityDB")) {
      pixelQualityFlags += MeasurementTracker::BadModules;
      if (pset.getUntrackedParameter<bool>("DebugPixelModuleQualityDB", false)) {
        pixelQualityDebugFlags += MeasurementTracker::BadModules;
      }
    }
    if (pset.getParameter<bool>("UsePixelROCQualityDB")) {
      pixelQualityFlags += MeasurementTracker::BadROCs;
      if (pset.getUntrackedParameter<bool>("DebugPixelROCQualityDB", false)) {
        pixelQualityDebugFlags += MeasurementTracker::BadROCs;
      }
    }

    return std::make_pair(pixelQualityFlags, pixelQualityDebugFlags);
  }
}  // namespace

MeasurementTrackerESProducer::MeasurementTrackerESProducer(const edm::ParameterSet &p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  auto c = setWhatProduced(this, myname);

  std::tie(pixelQualityFlags_, pixelQualityDebugFlags_) = pixelFlags(p);
  if (pixelQualityFlags_ != 0) {
    pixelQualityToken_ = c.consumes();
    pixelCablingToken_ = c.consumes();
  }

  std::tie(stripQualityFlags_, stripQualityDebugFlags_) = stripFlags(p);
  if (stripQualityFlags_ != 0) {
    stripQualityToken_ = c.consumes(edm::ESInputTag("", p.getParameter<std::string>("SiStripQualityLabel")));
    if (stripQualityFlags_ & MeasurementTrackerImpl::BadStrips) {
      auto makeBadStripCuts = [](edm::ParameterSet const &pset) {
        return StMeasurementConditionSet::BadStripCuts(pset.getParameter<uint32_t>("maxBad"),
                                                       pset.getParameter<uint32_t>("maxConsecutiveBad"));
      };

      auto cutPset = p.getParameter<edm::ParameterSet>("badStripCuts");
      badStripCuts_.tib = makeBadStripCuts(cutPset.getParameter<edm::ParameterSet>("TIB"));
      badStripCuts_.tob = makeBadStripCuts(cutPset.getParameter<edm::ParameterSet>("TOB"));
      badStripCuts_.tid = makeBadStripCuts(cutPset.getParameter<edm::ParameterSet>("TID"));
      badStripCuts_.tec = makeBadStripCuts(cutPset.getParameter<edm::ParameterSet>("TEC"));
    }
  }

  pixelCPEToken_ = c.consumes(edm::ESInputTag("", p.getParameter<std::string>("PixelCPE")));
  stripCPEToken_ = c.consumes(edm::ESInputTag("", p.getParameter<std::string>("StripCPE")));
  hitMatcherToken_ = c.consumes(edm::ESInputTag("", p.getParameter<std::string>("HitMatcher")));
  trackerTopologyToken_ = c.consumes();
  trackerGeomToken_ = c.consumes();
  geometricSearchTrackerToken_ = c.consumes();

  //FIXME:: just temporary solution for phase2!
  auto phase2 = p.getParameter<std::string>("Phase2StripCPE");
  if (not phase2.empty()) {
    usePhase2_ = true;
    phase2TrackerCPEToken_ = c.consumes(edm::ESInputTag("", phase2));
  }
}

MeasurementTrackerESProducer::~MeasurementTrackerESProducer() {}

std::unique_ptr<MeasurementTracker> MeasurementTrackerESProducer::produce(const CkfComponentsRecord &iRecord) {
  // ========= SiPixelQuality related tasks =============
  const SiPixelQuality *ptr_pixelQuality = nullptr;
  const SiPixelFedCabling *ptr_pixelCabling = nullptr;

  if (pixelQualityFlags_ != 0) {
    ptr_pixelQuality = &iRecord.get(pixelQualityToken_);
    ptr_pixelCabling = &iRecord.get(pixelCablingToken_);
  }

  // ========= SiStripQuality related tasks =============
  const SiStripQuality *ptr_stripQuality = nullptr;
  if (stripQualityFlags_ != 0) {
    ptr_stripQuality = &iRecord.get(stripQualityToken_);
  }

  const ClusterParameterEstimator<Phase2TrackerCluster1D> *ptr_phase2TrackerCPE = nullptr;
  if (usePhase2_) {
    ptr_phase2TrackerCPE = &iRecord.get(phase2TrackerCPEToken_);
  }
  return std::make_unique<MeasurementTrackerImpl>(badStripCuts_,
                                                  &iRecord.get(pixelCPEToken_),
                                                  &iRecord.get(stripCPEToken_),
                                                  &iRecord.get(hitMatcherToken_),
                                                  &iRecord.get(trackerTopologyToken_),
                                                  &iRecord.get(trackerGeomToken_),
                                                  &iRecord.get(geometricSearchTrackerToken_),
                                                  ptr_stripQuality,
                                                  stripQualityFlags_,
                                                  stripQualityDebugFlags_,
                                                  ptr_pixelQuality,
                                                  ptr_pixelCabling,
                                                  pixelQualityFlags_,
                                                  pixelQualityDebugFlags_,
                                                  ptr_phase2TrackerCPE);
}

void MeasurementTrackerESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("ComponentName", "");

  desc.add<std::string>("PixelCPE", "PixelCPEGeneric");
  desc.add<std::string>("StripCPE", "StripCPEfromTrackAngle");
  desc.add<std::string>("HitMatcher", "StandardMatcher");

  desc.add<std::string>("Phase2StripCPE", "")->setComment("empty string used to turn off Phase 2");

  desc.add<std::string>("SiStripQualityLabel", "");
  desc.add<bool>("UseStripModuleQualityDB", true);
  desc.addUntracked<bool>("DebugStripModuleQualityDB", false)->setComment("Dump out info on module status");
  desc.add<bool>("UseStripAPVFiberQualityDB", true)->setComment("Read APV and Fiber status from SiStripQuality");
  desc.addUntracked<bool>("DebugStripAPVFiberQualityDB", false)->setComment("Dump out info on module status");
  desc.add<bool>("MaskBadAPVFibers", true)
      ->setComment(
          "if set to true, clusters with barycenter on bad APV and Fibers are ignored. (UseStripAPVFiberQualityDB must "
          "also be true for this to work)");

  desc.add<bool>("UseStripStripQualityDB", true)->setComment("read Strip status from SiStripQuality");
  desc.addUntracked<bool>("DebugStripStripQualityDB", false)->setComment("dump out info on module status");

  {
    //used by MeasurementTrackerImpl
    edm::ParameterSetDescription badStripCutsDesc;
    badStripCutsDesc.add<uint32_t>("maxBad", 4);
    badStripCutsDesc.add<uint32_t>("maxConsecutiveBad", 2);

    edm::ParameterSetDescription mtiDesc;
    mtiDesc.add<edm::ParameterSetDescription>("TIB", badStripCutsDesc);
    mtiDesc.add<edm::ParameterSetDescription>("TOB", badStripCutsDesc);
    mtiDesc.add<edm::ParameterSetDescription>("TID", badStripCutsDesc);
    mtiDesc.add<edm::ParameterSetDescription>("TEC", badStripCutsDesc);

    desc.addOptional<edm::ParameterSetDescription>("badStripCuts", mtiDesc);
  }

  desc.add<bool>("UsePixelModuleQualityDB", true)->setComment("Use DB info at the module level (that is, detid level)");
  desc.addUntracked<bool>("DebugPixelModuleQualityDB", false)->setComment("dump out info on module status");
  desc.add<bool>("UsePixelROCQualityDB", true)->setComment("Use DB info at the ROC level");
  desc.addUntracked<bool>("DebugPixelROCQualityDB", false)->setComment("dump out info om module status ");

  descriptions.add("_MeasurementTrackerESProducer_default", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(MeasurementTrackerESProducer);
