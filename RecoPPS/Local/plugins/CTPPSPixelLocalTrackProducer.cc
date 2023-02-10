#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "RecoPPS/Local/interface/RPixDetPatternFinder.h"
#include "RecoPPS/Local/interface/RPixDetTrackFinder.h"

#include <memory>

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "RecoPPS/Local/interface/RPixRoadFinder.h"
#include "RecoPPS/Local/interface/RPixPlaneCombinatoryTracking.h"

#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"

namespace {
  constexpr int rocMask = 0xE000;
  constexpr int rocOffset = 13;
  constexpr int rocSizeInPixels = 4160;
}  // namespace

class CTPPSPixelLocalTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSPixelLocalTrackProducer(const edm::ParameterSet &parameterSet);

  ~CTPPSPixelLocalTrackProducer() override;

  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const int verbosity_;
  const int maxHitPerPlane_;
  const int maxHitPerRomanPot_;
  const int maxTrackPerRomanPot_;
  const int maxTrackPerPattern_;

  const edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> tokenCTPPSPixelRecHit_;
  const edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> tokenCTPPSGeometry_;
  edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher_;

  const edm::ESGetToken<CTPPSPixelAnalysisMask, CTPPSPixelAnalysisMaskRcd> tokenCTPPSPixelAnalysisMask_;

  const uint32_t numberOfPlanesPerPot_;

  std::unique_ptr<RPixDetPatternFinder> patternFinder_;
  std::unique_ptr<RPixDetTrackFinder> trackFinder_;
};

//------------------------------------------------------------------------------------------------//

CTPPSPixelLocalTrackProducer::CTPPSPixelLocalTrackProducer(const edm::ParameterSet &parameterSet)
    : verbosity_(parameterSet.getUntrackedParameter<int>("verbosity")),
      maxHitPerPlane_(parameterSet.getParameter<int>("maxHitPerPlane")),
      maxHitPerRomanPot_(parameterSet.getParameter<int>("maxHitPerRomanPot")),
      maxTrackPerRomanPot_(parameterSet.getParameter<int>("maxTrackPerRomanPot")),
      maxTrackPerPattern_(parameterSet.getParameter<int>("maxTrackPerPattern")),
      tokenCTPPSPixelRecHit_(
          consumes<edm::DetSetVector<CTPPSPixelRecHit>>(parameterSet.getParameter<edm::InputTag>("tag"))),
      tokenCTPPSGeometry_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord>()),
      tokenCTPPSPixelAnalysisMask_(esConsumes<CTPPSPixelAnalysisMask, CTPPSPixelAnalysisMaskRcd>()),
      numberOfPlanesPerPot_(parameterSet.getParameter<int>("numberOfPlanesPerPot")) {
  std::string patternFinderAlgorithm = parameterSet.getParameter<std::string>("patternFinderAlgorithm");
  std::string trackFitterAlgorithm = parameterSet.getParameter<std::string>("trackFinderAlgorithm");

  // pattern algorithm selector
  if (patternFinderAlgorithm == "RPixRoadFinder") {
    patternFinder_ = std::make_unique<RPixRoadFinder>(parameterSet);
  } else {
    throw cms::Exception("CTPPSPixelLocalTrackProducer")
        << "Pattern finder algorithm" << patternFinderAlgorithm << " does not exist";
  }

  std::vector<uint32_t> listOfAllPlanes;
  for (uint32_t i = 0; i < numberOfPlanesPerPot_; ++i) {
    listOfAllPlanes.push_back(i);
  }

  //tracking algorithm selector
  if (trackFitterAlgorithm == "RPixPlaneCombinatoryTracking") {
    trackFinder_ = std::make_unique<RPixPlaneCombinatoryTracking>(parameterSet);
  } else {
    throw cms::Exception("CTPPSPixelLocalTrackProducer")
        << "Tracking fitter algorithm" << trackFitterAlgorithm << " does not exist";
  }
  trackFinder_->setListOfPlanes(listOfAllPlanes);
  trackFinder_->initialize();
  produces<edm::DetSetVector<CTPPSPixelLocalTrack>>();
}

//------------------------------------------------------------------------------------------------//

CTPPSPixelLocalTrackProducer::~CTPPSPixelLocalTrackProducer() {}

//------------------------------------------------------------------------------------------------//

void CTPPSPixelLocalTrackProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("tag", edm::InputTag("ctppsPixelRecHits"))
      ->setComment("inputTag of the RecHits input for the tracking algorithm");
  desc.add<std::string>("patternFinderAlgorithm", "RPixRoadFinder")->setComment("algorithm type for pattern finder");
  desc.add<std::string>("trackFinderAlgorithm", "RPixPlaneCombinatoryTracking")
      ->setComment("algorithm type for track finder");
  desc.add<uint>("trackMinNumberOfPoints", 3)->setComment("minimum number of planes to produce a track");
  desc.addUntracked<int>("verbosity", 0)->setComment("verbosity for track producer");
  desc.add<double>("maximumChi2OverNDF", 5.)->setComment("maximum Chi2OverNDF for accepting the track");
  desc.add<double>("maximumXLocalDistanceFromTrack", 0.2)
      ->setComment("maximum x distance in mm to associate a point not used for fit to the track");
  desc.add<double>("maximumYLocalDistanceFromTrack", 0.3)
      ->setComment("maximum y distance in mm to associate a point not used for fit to the track");
  desc.add<int>("maxHitPerPlane", 20)
      ->setComment("maximum hits per plane, events with higher number will not be fitted");
  desc.add<int>("maxHitPerRomanPot", 60)
      ->setComment("maximum hits per roman pot, events with higher number will not be fitted");
  desc.add<int>("maxTrackPerRomanPot", 10)
      ->setComment("maximum tracks per roman pot, events with higher track will not be saved");
  desc.add<int>("maxTrackPerPattern", 5)
      ->setComment("maximum tracks per pattern, events with higher track will not be saved");
  desc.add<int>("numberOfPlanesPerPot", 6)->setComment("number of planes per pot");
  desc.add<double>("roadRadius", 1.0)->setComment("radius of pattern search window");
  desc.add<int>("minRoadSize", 3)->setComment("minimum number of points in a pattern");
  desc.add<int>("maxRoadSize", 20)->setComment("maximum number of points in a pattern");
  //parameter for 2-plane-pot reconstruction patch in run 3
  desc.add<double>("roadRadiusBadPot", 0.5)->setComment("radius of pattern search window for 2 plane pot");

  descriptions.add("ctppsPixelLocalTracks", desc);
}

//------------------------------------------------------------------------------------------------//

void CTPPSPixelLocalTrackProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Step A: get inputs

  edm::Handle<edm::DetSetVector<CTPPSPixelRecHit>> recHits;
  iEvent.getByToken(tokenCTPPSPixelRecHit_, recHits);
  edm::DetSetVector<CTPPSPixelRecHit> recHitVector(*recHits);

  // get geometry
  edm::ESHandle<CTPPSGeometry> geometryHandler = iSetup.getHandle(tokenCTPPSGeometry_);
  const CTPPSGeometry &geometry = *geometryHandler;
  geometryWatcher_.check(iSetup);

  // get mask

  // 2 planes pot flag vector 0=45-220, 1=45-210, 2=56-210, 3=56-220
  bool is2PlanePotV[4] = {false};

  if (!recHits->empty()) {
    const auto &mask = iSetup.getData(tokenCTPPSPixelAnalysisMask_);

    // Read Mask checking if 4 planes are masked and pot needs special treatment
    std::map<uint32_t, CTPPSPixelROCAnalysisMask> const &maschera = mask.analysisMask;

    // vector of masked rocs
    bool maskV[4][6][6] = {{{false}}};

    for (auto const &det : maschera) {
      CTPPSPixelDetId detId(det.first);
      unsigned int rocNum = (det.first & rocMask) >> rocOffset;
      if (rocNum > 5 || detId.plane() > 5)
        throw cms::Exception("InvalidRocOrPlaneNumber") << "roc number from mask: " << rocNum;

      if (detId.arm() == 0 && detId.rp() == 3) {
        if (detId.station() == 2) {                                                                // pot 45-220-far
          if (det.second.maskedPixels.size() == rocSizeInPixels || det.second.fullMask == true) {  // roc fully masked
            maskV[0][detId.plane()][rocNum] = true;
          }
        } else if (detId.station() == 0) {                                                         // pot 45-210-far
          if (det.second.maskedPixels.size() == rocSizeInPixels || det.second.fullMask == true) {  // roc fully masked
            maskV[1][detId.plane()][rocNum] = true;
          }
        }
      } else if (detId.arm() == 1 && detId.rp() == 3) {
        if (detId.station() == 0) {                                                                // pot 56-210-far
          if (det.second.maskedPixels.size() == rocSizeInPixels || det.second.fullMask == true) {  // roc fully masked
            maskV[2][detId.plane()][rocNum] = true;
          }
        } else if (detId.station() == 2) {                                                         // pot 56-220-far
          if (det.second.maskedPixels.size() == rocSizeInPixels || det.second.fullMask == true) {  // roc fully masked
            maskV[3][detId.plane()][rocNum] = true;
          }
        }
      }
    }
    // search for specific pattern that requires special reconstruction (use of two plane tracks)

    for (unsigned int i = 0; i < 4; i++) {
      unsigned int numberOfMaskedPlanes = 0;
      for (unsigned int j = 0; j < 6; j++) {
        if (maskV[i][j][0] && maskV[i][j][1] && maskV[i][j][4] && maskV[i][j][5])
          numberOfMaskedPlanes++;
      }
      if (numberOfMaskedPlanes == 4)
        is2PlanePotV[i] = true;  // search for exactly 4 planes fully masked
      edm::LogInfo("CTPPSPixelLocalTrackProducer") << " Masked planes in Pot #" << i << " = " << numberOfMaskedPlanes;
      if (is2PlanePotV[i])
        edm::LogInfo("CTPPSPixelLocalTrackProducer")
            << " Enabling 2 plane track reconstruction for pot #" << i << " (0=45-220, 1=45-210, 2=56-210, 3=56-220) ";
    }
  }
  std::vector<CTPPSPixelDetId> listOfPotWithHighOccupancyPlanes;
  std::map<CTPPSPixelDetId, uint32_t> mapHitPerPot;

  for (const auto &recHitSet : recHitVector) {
    if (verbosity_ > 2)
      edm::LogInfo("CTPPSPixelLocalTrackProducer")
          << "Hits found in plane = " << recHitSet.detId() << " number = " << recHitSet.size();
    CTPPSPixelDetId tmpRomanPotId = CTPPSPixelDetId(recHitSet.detId()).rpId();
    uint32_t hitOnPlane = recHitSet.size();

    //Get the number of hits per pot
    if (mapHitPerPot.find(tmpRomanPotId) == mapHitPerPot.end()) {
      mapHitPerPot[tmpRomanPotId] = hitOnPlane;
    } else
      mapHitPerPot[tmpRomanPotId] += hitOnPlane;

    //check is the plane occupancy is too high and save the corresponding pot
    if (maxHitPerPlane_ >= 0 && hitOnPlane > (uint32_t)maxHitPerPlane_) {
      if (verbosity_ > 2)
        edm::LogInfo("CTPPSPixelLocalTrackProducer")
            << " ---> Too many hits in the plane, pot will be excluded from tracking cleared";
      listOfPotWithHighOccupancyPlanes.push_back(tmpRomanPotId);
    }
  }

  //remove rechit for pot with too many hits or containing planes with too many hits
  for (const auto &recHitSet : recHitVector) {
    const auto tmpDetectorId = CTPPSPixelDetId(recHitSet.detId());
    const auto tmpRomanPotId = tmpDetectorId.rpId();

    if ((maxHitPerRomanPot_ >= 0 && mapHitPerPot[tmpRomanPotId] > (uint32_t)maxHitPerRomanPot_) ||
        find(listOfPotWithHighOccupancyPlanes.begin(), listOfPotWithHighOccupancyPlanes.end(), tmpRomanPotId) !=
            listOfPotWithHighOccupancyPlanes.end()) {
      edm::DetSet<CTPPSPixelRecHit> &tmpDetSet = recHitVector[tmpDetectorId];
      tmpDetSet.clear();
    }
  }

  edm::DetSetVector<CTPPSPixelLocalTrack> foundTracks;

  // Pattern finder

  patternFinder_->clear();
  patternFinder_->setHits(&recHitVector);
  patternFinder_->setGeometry(&geometry);
  patternFinder_->findPattern(is2PlanePotV);
  std::vector<RPixDetPatternFinder::Road> patternVector = patternFinder_->getPatterns();

  //loop on all the patterns
  int numberOfTracks = 0;

  for (const auto &pattern : patternVector) {
    CTPPSPixelDetId firstHitDetId = CTPPSPixelDetId(pattern.at(0).detId);
    CTPPSPixelDetId romanPotId = firstHitDetId.rpId();

    std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane>>
        hitOnPlaneMap;  //hit of the pattern organized by plane

    //loop on all the hits of the pattern
    for (const auto &hit : pattern) {
      CTPPSPixelDetId hitDetId = CTPPSPixelDetId(hit.detId);
      CTPPSPixelDetId tmpRomanPotId = hitDetId.rpId();

      if (tmpRomanPotId != romanPotId) {  //check that the hits belong to the same tracking station
        throw cms::Exception("CTPPSPixelLocalTrackProducer")
            << "Hits in the pattern must belong to the same tracking station";
      }

      if (hitOnPlaneMap.find(hitDetId) ==
          hitOnPlaneMap.end()) {  //add the plane key in case it is the first hit of that plane
        std::vector<RPixDetPatternFinder::PointInPlane> hitOnPlane;
        hitOnPlane.push_back(hit);
        hitOnPlaneMap[hitDetId] = hitOnPlane;
      } else
        hitOnPlaneMap[hitDetId].push_back(hit);  //add the hit to an existing plane key
    }

    trackFinder_->clear();
    trackFinder_->setRomanPotId(romanPotId);
    trackFinder_->setHits(&hitOnPlaneMap);
    trackFinder_->setGeometry(&geometry);
    trackFinder_->setZ0(geometry.rpTranslation(romanPotId).z());
    trackFinder_->findTracks(iEvent.getRun().id().run());
    std::vector<CTPPSPixelLocalTrack> tmpTracksVector = trackFinder_->getLocalTracks();

    if (verbosity_ > 2)
      edm::LogInfo("CTPPSPixelLocalTrackProducer") << "tmpTracksVector = " << tmpTracksVector.size();
    if (maxTrackPerPattern_ >= 0 && tmpTracksVector.size() > (uint32_t)maxTrackPerPattern_) {
      if (verbosity_ > 2)
        edm::LogInfo("CTPPSPixelLocalTrackProducer") << " ---> To many tracks in the pattern, cleared";
      continue;
    }

    for (const auto &track : tmpTracksVector) {
      ++numberOfTracks;
      edm::DetSet<CTPPSPixelLocalTrack> &tmpDetSet = foundTracks.find_or_insert(romanPotId);
      tmpDetSet.push_back(track);
    }
  }

  if (verbosity_ > 1)
    edm::LogInfo("CTPPSPixelLocalTrackProducer") << "Number of tracks will be saved = " << numberOfTracks;

  for (const auto &track : foundTracks) {
    if (verbosity_ > 1)
      edm::LogInfo("CTPPSPixelLocalTrackProducer")
          << "Track found in detId = " << track.detId() << " number = " << track.size();
    if (maxTrackPerRomanPot_ >= 0 && track.size() > (uint32_t)maxTrackPerRomanPot_) {
      if (verbosity_ > 1)
        edm::LogInfo("CTPPSPixelLocalTrackProducer") << " ---> Too many tracks in the pot, cleared";
      CTPPSPixelDetId tmpRomanPotId = CTPPSPixelDetId(track.detId());
      edm::DetSet<CTPPSPixelLocalTrack> &tmpDetSet = foundTracks[tmpRomanPotId];
      tmpDetSet.clear();
    }
  }

  iEvent.put(std::make_unique<edm::DetSetVector<CTPPSPixelLocalTrack>>(foundTracks));

  return;
}

DEFINE_FWK_MODULE(CTPPSPixelLocalTrackProducer);
