#include "L1Trigger/TrackFindingTracklet/interface/StubKiller.h"

using namespace std;

StubKiller::StubKiller()
    : killScenario_(0),
      trackerTopology_(nullptr),
      trackerGeometry_(nullptr),
      layersToKill_(vector<int>()),
      minPhiToKill_(0),
      maxPhiToKill_(0),
      minZToKill_(0),
      maxZToKill_(0),
      minRToKill_(0),
      maxRToKill_(0),
      fractionOfStubsToKillInLayers_(0),
      fractionOfStubsToKillEverywhere_(0),
      fractionOfModulesToKillEverywhere_(0) {}

void StubKiller::initialise(unsigned int killScenario,
                            const TrackerTopology* trackerTopology,
                            const TrackerGeometry* trackerGeometry) {
  killScenario_ = killScenario;
  trackerTopology_ = trackerTopology;
  trackerGeometry_ = trackerGeometry;

  switch (killScenario_) {
    // kill layer 5 in one quadrant + 5% random module loss to connect to what was done before
    case 1:
      layersToKill_ = {5};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 1000.0;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.05;
      break;

    // kill layer 1 in one quadrant + 5% random module loss
    case 2:
      layersToKill_ = {1};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 1000.0;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.05;
      break;

    // kill layer 1 + layer 2, both in same quadrant
    case 3:
      layersToKill_ = {1, 2};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 1000.0;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0;
      break;

    // kill layer 1 and disk 1, both in same quadrant
    case 4:
      layersToKill_ = {1, 11};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 66.5;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0;
      break;

    // 5% random module loss throughout tracker
    case 5:
      layersToKill_ = {};
      fractionOfStubsToKillInLayers_ = 0;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.05;
      break;

    // 1% random module loss throughout tracker
    case 6:
      layersToKill_ = {};
      fractionOfStubsToKillInLayers_ = 0;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.01;
      break;

    // kill layer 5 in one quadrant + 1% random module loss
    case 7:
      layersToKill_ = {5};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 1000.0;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.01;
      break;

    // kill layer 1 in one quadrant +1 % random module loss
    case 8:
      layersToKill_ = {1};
      minPhiToKill_ = 0.0;
      maxPhiToKill_ = TMath::PiOver2();
      minZToKill_ = -1000.0;
      maxZToKill_ = 0.0;
      minRToKill_ = 0.0;
      maxRToKill_ = 1000.0;
      fractionOfStubsToKillInLayers_ = 1;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.01;
      break;

    // 10% random module loss throughout tracker
    case 9:
      layersToKill_ = {};
      fractionOfStubsToKillInLayers_ = 0;
      fractionOfStubsToKillEverywhere_ = 0;
      fractionOfModulesToKillEverywhere_ = 0.10;
      break;
  }

  deadModules_.clear();
  if (fractionOfModulesToKillEverywhere_ > 0) {
    this->chooseModulesToKill();
  }
  this->addDeadLayerModulesToDeadModuleList();
}

void StubKiller::chooseModulesToKill() {
  TRandom3* randomGenerator = new TRandom3();
  randomGenerator->SetSeed(0);

  for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
    if (!trackerTopology_->isLower(gd->geographicalId()))
      continue;
    if (randomGenerator->Uniform(0.0, 1.0) < fractionOfModulesToKillEverywhere_) {
      deadModules_[gd->geographicalId()] = 1;
    }
  }
}

void StubKiller::addDeadLayerModulesToDeadModuleList() {
  for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
    float moduleR = gd->position().perp();
    float moduleZ = gd->position().z();
    float modulePhi = gd->position().phi();
    DetId geoDetId = gd->geographicalId();
    bool isInBarrel = geoDetId.subdetId() == StripSubdetector::TOB || geoDetId.subdetId() == StripSubdetector::TIB;

    int layerID = 0;
    if (isInBarrel) {
      layerID = trackerTopology_->layer(geoDetId);
    } else {
      layerID = 10 * trackerTopology_->side(geoDetId) + trackerTopology_->tidWheel(geoDetId);
    }

    if (find(layersToKill_.begin(), layersToKill_.end(), layerID) != layersToKill_.end()) {
      if (modulePhi < -1.0 * TMath::Pi())
        modulePhi += 2.0 * TMath::Pi();
      else if (modulePhi > TMath::Pi())
        modulePhi -= 2.0 * TMath::Pi();

      if (modulePhi > minPhiToKill_ && modulePhi < maxPhiToKill_ && moduleZ > minZToKill_ && moduleZ < maxZToKill_ &&
          moduleR > minRToKill_ && moduleR < maxRToKill_) {
        if (deadModules_.find(gd->geographicalId()) == deadModules_.end()) {
          deadModules_[gd->geographicalId()] = fractionOfStubsToKillInLayers_;
        }
      }
    }
  }
}

bool StubKiller::killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub) {
  if (killScenario_ == 0)
    return false;
  else {
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
    bool killStubInDeadModules = killStubInDeadModule(stub);
    return killStubRandomly || killStubInDeadModules;
  }
}

// layersToKill - a vector stating the layers we are killing stubs in.  Can be an empty vector.
// Barrel layers are encoded as 1-6. The endcap layers are encoded as 11-15 (-z) and 21-25 (+z)
// min/max Phi/Z/R - stubs within the region specified by these boundaries and layersToKill are flagged for killing
// fractionOfStubsToKillInLayers - The fraction of stubs to kill in the specified layers/region.
// fractionOfStubsToKillEverywhere - The fraction of stubs to kill throughout the tracker
bool StubKiller::killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub,
                          const vector<int> layersToKill,
                          const double minPhiToKill,
                          const double maxPhiToKill,
                          const double minZToKill,
                          const double maxZToKill,
                          const double minRToKill,
                          const double maxRToKill,
                          const double fractionOfStubsToKillInLayers,
                          const double fractionOfStubsToKillEverywhere) {
  // Only kill stubs in specified layers
  if (layersToKill.empty()) {
    // Get the layer the stub is in, and check if it's in the layer you want to kill
    DetId stackDetid = stub->getDetId();
    DetId geoDetId(stackDetid.rawId() + 1);

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

      // Just in case phi is outside of -pi -> pi
      double stubPhi = pos.phi();
      if (stubPhi < -1.0 * TMath::Pi())
        stubPhi += 2.0 * TMath::Pi();
      else if (stubPhi > TMath::Pi())
        stubPhi -= 2.0 * TMath::Pi();

      if (stubPhi > minPhiToKill && stubPhi < maxPhiToKill && pos.z() > minZToKill && pos.z() < maxZToKill &&
          pos.perp() > minRToKill && pos.perp() < maxRToKill) {
        // Kill fraction of stubs
        if (fractionOfStubsToKillInLayers == 1) {
          return true;
        } else {
          static TRandom randomGenerator;
          if (randomGenerator.Rndm() < fractionOfStubsToKillInLayers) {
            return true;
          }
        }
      }
    }
  }

  // Kill fraction of stubs throughout tracker
  if (fractionOfStubsToKillEverywhere > 0) {
    static TRandom randomGenerator;
    if (randomGenerator.Rndm() < fractionOfStubsToKillEverywhere) {
      return true;
    }
  }

  return false;
}

bool StubKiller::killStubInDeadModule(const TTStub<Ref_Phase2TrackerDigi_>* stub) {
  if (deadModules_.empty()) {
    DetId stackDetid = stub->getDetId();
    DetId geoDetId(stackDetid.rawId() + 1);
    if (deadModules_.find(geoDetId) != deadModules_.end())
      return true;
  }

  return false;
}
