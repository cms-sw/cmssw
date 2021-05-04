#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

namespace tmtt {

  StubKiller::StubKiller(StubKiller::KillOptions killScenario,
                         const TrackerTopology* trackerTopology,
                         const TrackerGeometry* trackerGeometry,
                         const edm::Event& event)
      : killScenario_(killScenario),
        trackerTopology_(trackerTopology),
        trackerGeometry_(trackerGeometry),
        minPhiToKill_(0),
        maxPhiToKill_(0),
        minZToKill_(0),
        maxZToKill_(0),
        minRToKill_(0),
        maxRToKill_(0),
        fractionOfStubsToKillInLayers_(0),
        fractionOfStubsToKillEverywhere_(0),
        fractionOfModulesToKillEverywhere_(0) {
    if (rndmService_.isAvailable()) {
      rndmEngine_ = &(rndmService_->getEngine(event.streamID()));
    } else {
      throw cms::Exception("BadConfig")
          << "StubKiller: requires RandomNumberGeneratorService, not present in cfg file, namely:" << endl
          << "process.RandomNumberGeneratorService=cms.Service('RandomNumberGeneratorService',TMTrackProducer=cms.PSet("
             "initialSeed=cms.untracked.uint32(12345)))";
    }

    // These scenarios correspond to slide 12 of  https://indico.cern.ch/event/719985/contributions/2970687/attachments/1634587/2607365/StressTestTF-Acosta-Apr18.pdf
    // Scenario 1

    // kill layer 5 in one quadrant +5 % random module loss to connect to what was done before
    if (killScenario_ == KillOptions::layer5) {
      layersToKill_ = {5};
      minPhiToKill_ = 0;
      maxPhiToKill_ = 0.5 * M_PI;
      minZToKill_ = -1000;
      maxZToKill_ = 0;
      minRToKill_ = 0;
      maxRToKill_ = 1000;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.05;
    }
    // Scenario 2
    // kill layer 1 in one quadrant +5 % random module loss
    else if (killScenario_ == KillOptions::layer1) {
      layersToKill_ = {1};
      minPhiToKill_ = 0;
      maxPhiToKill_ = 0.5 * M_PI;
      minZToKill_ = -1000;
      maxZToKill_ = 0;
      minRToKill_ = 0;
      maxRToKill_ = 1000;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.05;
    }
    // Scenario 3
    // kill layer 1 + layer 2, both in same quadrant
    else if (killScenario_ == KillOptions::layer1layer2) {
      layersToKill_ = {1, 2};
      minPhiToKill_ = 0;
      maxPhiToKill_ = 0.5 * M_PI;
      minZToKill_ = -1000;
      maxZToKill_ = 0;
      minRToKill_ = 0;
      maxRToKill_ = 1000;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0;
    }
    // Scenario 4
    // kill layer 1 and disk 1, both in same quadrant
    else if (killScenario_ == KillOptions::layer1disk1) {
      layersToKill_ = {1, 11};
      minPhiToKill_ = 0;
      maxPhiToKill_ = 0.5 * M_PI;
      minZToKill_ = -1000;
      maxZToKill_ = 0;
      minRToKill_ = 0;
      maxRToKill_ = 66.5;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0;
    }
    // An extra scenario not listed in the slides
    // 5% random module loss throughout tracker
    else if (killScenario_ == KillOptions::random) {
      layersToKill_ = {};
      fractionOfStubsToKillInLayers_ = 0;
      fractionOfStubsToKillEverywhere_ = 0.;
      fractionOfModulesToKillEverywhere_ = 0.05;
    }

    deadModules_.clear();
    if (fractionOfModulesToKillEverywhere_ > 0) {
      this->chooseModulesToKill();
    }
    this->addDeadLayerModulesToDeadModuleList();
  }

  // Indicate if given stub was killed by dead tracker module, based on dead module scenario.

  bool StubKiller::killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub) const {
    if (killScenario_ == KillOptions::none)
      return false;
    else {
      // Check if stub is in dead region specified by *ToKill_
      // And randomly kill stubs throughout tracker (not just thos in specific regions/modules)
      bool killStubRandomly = killStub(stub,
                                       layersToKill_,
                                       minPhiToKill_,
                                       maxPhiToKill_,
                                       minZToKill_,
                                       maxZToKill_,
                                       minRToKill_,
                                       maxRToKill_,
                                       fractionOfStubsToKillInLayers_,
                                       fractionOfStubsToKillEverywhere_);
      // Kill modules in specifid modules
      // Random modules throughout the tracker, and those modules in specific regions (so may already have been killed by killStub above)
      bool killStubInDeadModules = killStubInDeadModule(stub);
      return killStubRandomly || killStubInDeadModules;
    }
  }

  // Indicate if given stub was killed by dead tracker module, based on specified dead regions
  // rather than based on the dead module scenario.
  // layersToKill - a vector stating the layers we are killing stubs in.  Can be an empty vector.
  // Barrel layers are encoded as 1-6. The endcap layers are encoded as 11-15 (-z) and 21-25 (+z)
  // min/max Phi/Z/R - stubs within the region specified by these boundaries and layersToKill are flagged for killing
  // fractionOfStubsToKillInLayers - The fraction of stubs to kill in the specified layers/region.
  // fractionOfStubsToKillEverywhere - The fraction of stubs to kill throughout the tracker

  bool StubKiller::killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub,
                            const vector<int>& layersToKill,
                            const double minPhiToKill,
                            const double maxPhiToKill,
                            const double minZToKill,
                            const double maxZToKill,
                            const double minRToKill,
                            const double maxRToKill,
                            const double fractionOfStubsToKillInLayers,
                            const double fractionOfStubsToKillEverywhere) const {
    // Only kill stubs in specified layers
    if (not layersToKill.empty()) {
      // Get the layer the stub is in, and check if it's in the layer you want to kill
      DetId stackDetid = stub->getDetId();
      DetId geoDetId(stackDetid.rawId() + 1);

      // If this module is in the deadModule list, don't also try to kill the stub here
      if (deadModules_.empty() || deadModules_.find(geoDetId) == deadModules_.end()) {
        bool isInBarrel = geoDetId.subdetId() == StripSubdetector::TOB || geoDetId.subdetId() == StripSubdetector::TIB;

        int layerID = 0;
        if (isInBarrel) {
          layerID = trackerTopology_->layer(geoDetId);
        } else {
          layerID = 10 * trackerTopology_->side(geoDetId) + trackerTopology_->tidWheel(geoDetId);
        }

        if (find(layersToKill.begin(), layersToKill.end(), layerID) != layersToKill.end()) {
          // Get the phi and z of stub, and check if it's in the region you want to kill
          const GeomDetUnit* det0 = trackerGeometry_->idToDetUnit(geoDetId);
          const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(det0);
          const PixelTopology* topol = dynamic_cast<const PixelTopology*>(&(theGeomDet->specificTopology()));
          MeasurementPoint measurementPoint = stub->clusterRef(0)->findAverageLocalCoordinatesCentered();
          LocalPoint clustlp = topol->localPosition(measurementPoint);
          GlobalPoint pos = theGeomDet->surface().toGlobal(clustlp);

          double stubPhi = reco::deltaPhi(pos.phi(), 0.);

          if (stubPhi > minPhiToKill && stubPhi < maxPhiToKill && pos.z() > minZToKill && pos.z() < maxZToKill &&
              pos.perp() > minRToKill && pos.perp() < maxRToKill) {
            // Kill fraction of stubs
            if (fractionOfStubsToKillInLayers == 1) {
              return true;
            } else {
              if (rndmEngine_->flat() < fractionOfStubsToKillInLayers) {
                return true;
              }
            }
          }
        }
      }
    }

    // Kill fraction of stubs throughout tracker
    if (fractionOfStubsToKillEverywhere > 0) {
      if (rndmEngine_->flat() < fractionOfStubsToKillEverywhere) {
        return true;
      }
    }

    return false;
  }

  // Indicate if given stub was in (partially) dead tracker module, based on dead module scenario.

  bool StubKiller::killStubInDeadModule(const TTStub<Ref_Phase2TrackerDigi_>* stub) const {
    if (not deadModules_.empty()) {
      DetId stackDetid = stub->getDetId();
      DetId geoDetId(stackDetid.rawId() + 1);
      auto deadModule = deadModules_.find(geoDetId);
      if (deadModule != deadModules_.end()) {
        if (deadModule->second == 1) {
          return true;
        } else {
          if (rndmEngine_->flat() < deadModule->second) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // Identify modules to be killed, chosen randomly from those in the whole tracker.

  void StubKiller::chooseModulesToKill() {
    for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
      if (!trackerTopology_->isLower(gd->geographicalId()))
        continue;
      if (rndmEngine_->flat() < fractionOfModulesToKillEverywhere_) {
        deadModules_[gd->geographicalId()] = 1;
      }
    }
  }

  //  Identify modules to be killed, chosen based on location in tracker.

  void StubKiller::addDeadLayerModulesToDeadModuleList() {
    for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
      float moduleR = gd->position().perp();
      float moduleZ = gd->position().z();
      float modulePhi = reco::deltaPhi(gd->position().phi(), 0.);
      DetId geoDetId = gd->geographicalId();
      bool isInBarrel = geoDetId.subdetId() == StripSubdetector::TOB || geoDetId.subdetId() == StripSubdetector::TIB;

      int layerID = 0;
      if (isInBarrel) {
        layerID = trackerTopology_->layer(geoDetId);
      } else {
        layerID = 10 * trackerTopology_->side(geoDetId) + trackerTopology_->tidWheel(geoDetId);
      }
      if (find(layersToKill_.begin(), layersToKill_.end(), layerID) != layersToKill_.end()) {
        if (modulePhi > minPhiToKill_ && modulePhi < maxPhiToKill_ && moduleZ > minZToKill_ && moduleZ < maxZToKill_ &&
            moduleR > minRToKill_ && moduleR < maxRToKill_) {
          if (deadModules_.find(gd->geographicalId()) == deadModules_.end()) {
            deadModules_[gd->geographicalId()] = fractionOfStubsToKillInLayers_;
          }
        }
      }
    }
  }
};  // namespace tmtt
