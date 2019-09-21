// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      ElectronSeedGenerator.
//
/**\class ElectronSeedGenerator EgammaElectronAlgos/ElectronSeedGenerator

 Description: Top algorithm producing ElectronSeeds, ported from ORCA

 Implementation:
     future redesign...
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <utility>

ElectronSeedGenerator::ElectronSeedGenerator(const edm::ParameterSet &pset, const ElectronSeedGenerator::Tokens &ts)
    : dynamicPhiRoad_(pset.getParameter<bool>("dynamicPhiRoad")),
      verticesTag_(ts.token_vtx),
      beamSpotTag_(ts.token_bs),
      lowPtThresh_(pset.getParameter<double>("LowPtThreshold")),
      highPtThresh_(pset.getParameter<double>("HighPtThreshold")),
      nSigmasDeltaZ1_(pset.getParameter<double>("nSigmasDeltaZ1")),
      deltaZ1WithVertex_(pset.getParameter<double>("deltaZ1WithVertex")),
      sizeWindowENeg_(pset.getParameter<double>("SizeWindowENeg")),
      deltaPhi1Low_(pset.getParameter<double>("DeltaPhi1Low")),
      deltaPhi1High_(pset.getParameter<double>("DeltaPhi1High")),
      // so that deltaPhi1 = dPhi1Coef1_ + dPhi1Coef2_/clusterEnergyT
      dPhi1Coef2_(dynamicPhiRoad_ ? (deltaPhi1Low_ - deltaPhi1High_) / (1. / lowPtThresh_ - 1. / highPtThresh_) : 0.),
      dPhi1Coef1_(dynamicPhiRoad_ ? deltaPhi1Low_ - dPhi1Coef2_ / lowPtThresh_ : 0.),
      propagator_(nullptr),
      measurementTracker_(nullptr),
      measurementTrackerEventTag_(ts.token_measTrkEvt),
      setup_(nullptr),
      measurementTrackerName_(pset.getParameter<std::string>("measurementTrackerName")),
      // use of reco vertex
      useRecoVertex_(pset.getParameter<bool>("useRecoVertex")),
      // new B/F configurables
      deltaPhi2B_(pset.getParameter<double>("DeltaPhi2B")),
      deltaPhi2F_(pset.getParameter<double>("DeltaPhi2F")),
      phiMin2B_(pset.getParameter<double>("PhiMin2B")),
      phiMin2F_(pset.getParameter<double>("PhiMin2F")),
      phiMax2B_(pset.getParameter<double>("PhiMax2B")),
      phiMax2F_(pset.getParameter<double>("PhiMax2F")),
      electronMatcher_(pset.getParameter<double>("ePhiMin1"),
                       pset.getParameter<double>("ePhiMax1"),
                       phiMin2B_,
                       phiMax2B_,
                       phiMin2F_,
                       phiMax2F_,
                       pset.getParameter<double>("z2MinB"),
                       pset.getParameter<double>("z2MaxB"),
                       pset.getParameter<double>("r2MinF"),
                       pset.getParameter<double>("r2MaxF"),
                       pset.getParameter<double>("rMinI"),
                       pset.getParameter<double>("rMaxI"),
                       pset.getParameter<bool>("searchInTIDTEC")),
      positronMatcher_(pset.getParameter<double>("pPhiMin1"),
                       pset.getParameter<double>("pPhiMax1"),
                       phiMin2B_,
                       phiMax2B_,
                       phiMin2F_,
                       phiMax2F_,
                       pset.getParameter<double>("z2MinB"),
                       pset.getParameter<double>("z2MaxB"),
                       pset.getParameter<double>("r2MinF"),
                       pset.getParameter<double>("r2MaxF"),
                       pset.getParameter<double>("rMinI"),
                       pset.getParameter<double>("rMaxI"),
                       pset.getParameter<bool>("searchInTIDTEC")) {
  if (!pset.getParameter<bool>("fromTrackerSeeds")) {
    throw cms::Exception("NotSupported")
        << "Setting the fromTrackerSeeds parameter in ElectronSeedGenerator to True is not supported anymore.\n";
  }
}

void ElectronSeedGenerator::setupES(const edm::EventSetup &setup) {
  // get records if necessary (called once per event)
  bool tochange = false;

  if (cacheIDMagField_ != setup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    setup.get<IdealMagneticFieldRecord>().get(magField_);
    cacheIDMagField_ = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    propagator_ = std::make_unique<PropagatorWithMaterial>(alongMomentum, .000511, &(*magField_));
    tochange = true;
  }

  if (cacheIDTrkGeom_ != setup.get<TrackerDigiGeometryRecord>().cacheIdentifier()) {
    cacheIDTrkGeom_ = setup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
    setup.get<TrackerDigiGeometryRecord>().get(trackerGeometry_);
    tochange = true;  //FIXME
  }

  if (tochange) {
    electronMatcher_.setES(magField_.product(), measurementTracker_, trackerGeometry_.product());
    positronMatcher_.setES(magField_.product(), measurementTracker_, trackerGeometry_.product());
  }

  if (cacheIDNavSchool_ != setup.get<NavigationSchoolRecord>().cacheIdentifier()) {
    edm::ESHandle<NavigationSchool> nav;
    setup.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
    cacheIDNavSchool_ = setup.get<NavigationSchoolRecord>().cacheIdentifier();
    navigationSchool_ = nav.product();
  }
}

bool equivalent(const TrajectorySeed &s1, const TrajectorySeed &s2) {
  if (s1.nHits() != s2.nHits())
    return false;

  unsigned int nHits;
  TrajectorySeed::range r1 = s1.recHits(), r2 = s2.recHits();
  TrajectorySeed::const_iterator i1, i2;
  for (i1 = r1.first, i2 = r2.first, nHits = 0; i1 != r1.second; ++i1, ++i2, ++nHits) {
    if (!i1->isValid() || !i2->isValid())
      return false;
    if (i1->geographicalId() != i2->geographicalId())
      return false;
    if (!(i1->localPosition() == i2->localPosition()))
      return false;
  }

  return true;
}

void ElectronSeedGenerator::run(edm::Event &e,
                                const edm::EventSetup &setup,
                                const reco::SuperClusterRefVector &sclRefs,
                                const std::vector<float> &hoe1s,
                                const std::vector<float> &hoe2s,
                                const std::vector<const TrajectorySeedCollection *> &seedsV,
                                reco::ElectronSeedCollection &out) {
  initialSeedCollectionVector_ = &seedsV;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo = tTopoHand.product();

  setup_ = &setup;

  // Step A: set Event for the TrajectoryBuilder
  auto const &data = e.get(measurementTrackerEventTag_);
  electronMatcher_.setEvent(data);
  positronMatcher_.setEvent(data);

  // get the beamspot from the Event:
  //e.getByType(beamSpot_);
  e.getByToken(beamSpotTag_, beamSpot_);

  // if required get the vertices
  if (useRecoVertex_)
    e.getByToken(verticesTag_, vertices_);

  for (unsigned int i = 0; i < sclRefs.size(); ++i) {
    // Find the seeds
    recHits_.clear();

    LogDebug("ElectronSeedGenerator") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(sclRefs[i], hoe1s[i], hoe2s[i], out, tTopo);
  }

  LogDebug("ElectronSeedGenerator") << ": For event " << e.id();
  LogDebug("ElectronSeedGenerator") << "Nr of superclusters after filter: " << sclRefs.size()
                                    << ", no. of ElectronSeeds found  = " << out.size();
}

void ElectronSeedGenerator::seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
                                                 float hoe1,
                                                 float hoe2,
                                                 reco::ElectronSeedCollection &out,
                                                 const TrackerTopology *tTopo) {
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(), seedCluster->position().y(), seedCluster->position().z());
  reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster);

  if (dynamicPhiRoad_) {
    float clusterEnergyT = clusterEnergy / cosh(EleRelPoint(clusterPos, beamSpot_->position()).eta());

    float deltaPhi1;
    if (clusterEnergyT < lowPtThresh_) {
      deltaPhi1 = deltaPhi1Low_;
    } else if (clusterEnergyT > highPtThresh_) {
      deltaPhi1 = deltaPhi1High_;
    } else {
      deltaPhi1 = dPhi1Coef1_ + dPhi1Coef2_ / clusterEnergyT;
    }

    float ephimin1 = -deltaPhi1 * sizeWindowENeg_;
    float ephimax1 = deltaPhi1 * (1. - sizeWindowENeg_);
    float pphimin1 = -deltaPhi1 * (1. - sizeWindowENeg_);
    float pphimax1 = deltaPhi1 * sizeWindowENeg_;

    float phimin2B = -deltaPhi2B_ / 2.;
    float phimax2B = deltaPhi2B_ / 2.;
    float phimin2F = -deltaPhi2F_ / 2.;
    float phimax2F = deltaPhi2F_ / 2.;

    electronMatcher_.set1stLayer(ephimin1, ephimax1);
    positronMatcher_.set1stLayer(pphimin1, pphimax1);
    electronMatcher_.set2ndLayer(phimin2B, phimax2B, phimin2F, phimax2F);
    positronMatcher_.set2ndLayer(phimin2B, phimax2B, phimin2F, phimax2F);
  }

  if (!useRecoVertex_)  // here use the beam spot position
  {
    double sigmaZ = beamSpot_->sigmaZ();
    double sigmaZ0Error = beamSpot_->sigmaZ0Error();
    double sq = sqrt(sigmaZ * sigmaZ + sigmaZ0Error * sigmaZ0Error);
    double myZmin1 = beamSpot_->position().z() - nSigmasDeltaZ1_ * sq;
    double myZmax1 = beamSpot_->position().z() + nSigmasDeltaZ1_ * sq;

    GlobalPoint vertexPos;
    ele_convert(beamSpot_->position(), vertexPos);

    electronMatcher_.set1stLayerZRange(myZmin1, myZmax1);
    positronMatcher_.set1stLayerZRange(myZmin1, myZmax1);

    // try electron
    auto elePixelSeeds =
        electronMatcher_.compatibleSeeds(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, -1.);
    seedsFromTrajectorySeeds(elePixelSeeds, caloCluster, hoe1, hoe2, out, false);
    // try positron
    auto posPixelSeeds =
        positronMatcher_.compatibleSeeds(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, 1.);
    seedsFromTrajectorySeeds(posPixelSeeds, caloCluster, hoe1, hoe2, out, true);

  } else  // here we use the reco vertices
  {
    electronMatcher_.setUseRecoVertex(true);  //Hit matchers need to know that the vertex is known
    positronMatcher_.setUseRecoVertex(true);

    const std::vector<reco::Vertex> *vtxCollection = vertices_.product();
    std::vector<reco::Vertex>::const_iterator vtxIter;
    for (vtxIter = vtxCollection->begin(); vtxIter != vtxCollection->end(); vtxIter++) {
      GlobalPoint vertexPos(vtxIter->position().x(), vtxIter->position().y(), vtxIter->position().z());
      double myZmin1, myZmax1;
      if (vertexPos.z() == beamSpot_->position().z()) {  // in case vetex not found
        double sigmaZ = beamSpot_->sigmaZ();
        double sigmaZ0Error = beamSpot_->sigmaZ0Error();
        double sq = sqrt(sigmaZ * sigmaZ + sigmaZ0Error * sigmaZ0Error);
        myZmin1 = beamSpot_->position().z() - nSigmasDeltaZ1_ * sq;
        myZmax1 = beamSpot_->position().z() + nSigmasDeltaZ1_ * sq;
      } else {  // a vertex has been recoed
        myZmin1 = vtxIter->position().z() - deltaZ1WithVertex_;
        myZmax1 = vtxIter->position().z() + deltaZ1WithVertex_;
      }

      electronMatcher_.set1stLayerZRange(myZmin1, myZmax1);
      positronMatcher_.set1stLayerZRange(myZmin1, myZmax1);

      // try electron
      auto elePixelSeeds =
          electronMatcher_.compatibleSeeds(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, -1.);
      seedsFromTrajectorySeeds(elePixelSeeds, caloCluster, hoe1, hoe2, out, false);
      // try positron
      auto posPixelSeeds =
          positronMatcher_.compatibleSeeds(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, 1.);
      seedsFromTrajectorySeeds(posPixelSeeds, caloCluster, hoe1, hoe2, out, true);
    }
  }

  return;
}

void ElectronSeedGenerator::seedsFromRecHits(std::vector<std::pair<RecHitWithDist, ConstRecHitPointer> > &pixelHits,
                                             PropagationDirection &dir,
                                             const GlobalPoint &vertexPos,
                                             const reco::ElectronSeed::CaloClusterRef &cluster,
                                             reco::ElectronSeedCollection &out,
                                             bool positron) {
  if (!pixelHits.empty()) {
    LogDebug("ElectronSeedGenerator") << "Compatible " << (positron ? "positron" : "electron") << " hits found.";
  }

  for (auto &v : pixelHits) {
    if (!positron) {
      v.first.invert();
    }
    if (!prepareElTrackSeed(v.first.recHit(), v.second, vertexPos)) {
      continue;
    }
    reco::ElectronSeed seed(pts_, recHits_, dir);
    seed.setCaloCluster(cluster);
    addSeed(seed, nullptr, positron, out);
  }
}

void ElectronSeedGenerator::seedsFromTrajectorySeeds(const std::vector<SeedWithInfo> &pixelSeeds,
                                                     const reco::ElectronSeed::CaloClusterRef &cluster,
                                                     float hoe1,
                                                     float hoe2,
                                                     reco::ElectronSeedCollection &out,
                                                     bool positron) {
  if (!pixelSeeds.empty()) {
    LogDebug("ElectronSeedGenerator") << "Compatible " << (positron ? "positron" : "electron") << " seeds found.";
  }

  std::vector<SeedWithInfo>::const_iterator s;
  for (s = pixelSeeds.begin(); s != pixelSeeds.end(); s++) {
    reco::ElectronSeed seed(s->seed());
    seed.setCaloCluster(cluster);
    seed.initTwoHitSeed(s->hitsMask());
    addSeed(seed, &*s, positron, out);
  }
}

void ElectronSeedGenerator::addSeed(reco::ElectronSeed &seed,
                                    const SeedWithInfo *info,
                                    bool positron,
                                    reco::ElectronSeedCollection &out) {
  if (!info) {
    out.emplace_back(seed);
    return;
  }

  if (positron) {
    seed.setPosAttributes(info->dRz2(), info->dPhi2(), info->dRz1(), info->dPhi1());
  } else {
    seed.setNegAttributes(info->dRz2(), info->dPhi2(), info->dRz1(), info->dPhi1());
  }
  for (auto resItr = out.begin(); resItr != out.end(); ++resItr) {
    if ((seed.caloCluster().key() == resItr->caloCluster().key()) && (seed.hitsMask() == resItr->hitsMask()) &&
        equivalent(seed, *resItr)) {
      if (positron) {
        if (resItr->dRz2Pos() == std::numeric_limits<float>::infinity() &&
            resItr->dRz2() != std::numeric_limits<float>::infinity()) {
          resItr->setPosAttributes(info->dRz2(), info->dPhi2(), info->dRz1(), info->dPhi1());
          seed.setNegAttributes(resItr->dRz2(), resItr->dPhi2(), resItr->dRz1(), resItr->dPhi1());
          break;
        } else {
          if (resItr->dRz2Pos() != std::numeric_limits<float>::infinity()) {
            if (resItr->dRz2Pos() != seed.dRz2Pos()) {
              edm::LogWarning("ElectronSeedGenerator|BadValue")
                  << "this similar old seed already has another dRz2Pos"
                  << "\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)resItr->hitsMask() << "/"
                  << resItr->dRz2() << "/" << resItr->dPhi2() << "/" << resItr->dRz2Pos() << "/" << resItr->dPhi2Pos()
                  << "\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)seed.hitsMask() << "/"
                  << seed.dRz2() << "/" << seed.dPhi2() << "/" << seed.dRz2Pos() << "/" << seed.dPhi2Pos();
            }
          }
        }
      } else {
        if (resItr->dRz2() == std::numeric_limits<float>::infinity() &&
            resItr->dRz2Pos() != std::numeric_limits<float>::infinity()) {
          resItr->setNegAttributes(info->dRz2(), info->dPhi2(), info->dRz1(), info->dPhi1());
          seed.setPosAttributes(resItr->dRz2Pos(), resItr->dPhi2Pos(), resItr->dRz1Pos(), resItr->dPhi1Pos());
          break;
        } else {
          if (resItr->dRz2() != std::numeric_limits<float>::infinity()) {
            if (resItr->dRz2() != seed.dRz2()) {
              edm::LogWarning("ElectronSeedGenerator|BadValue")
                  << "this old seed already has another dRz2"
                  << "\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)resItr->hitsMask() << "/"
                  << resItr->dRz2() << "/" << resItr->dPhi2() << "/" << resItr->dRz2Pos() << "/" << resItr->dPhi2Pos()
                  << "\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)seed.hitsMask() << "/"
                  << seed.dRz2() << "/" << seed.dPhi2() << "/" << seed.dRz2Pos() << "/" << seed.dPhi2Pos();
            }
          }
        }
      }
    }
  }

  out.emplace_back(seed);
}

bool ElectronSeedGenerator::prepareElTrackSeed(ConstRecHitPointer innerhit,
                                               ConstRecHitPointer outerhit,
                                               const GlobalPoint &vertexPos) {
  // debug prints
  LogDebug("") << "[ElectronSeedGenerator::prepareElTrackSeed] inner PixelHit   x,y,z " << innerhit->globalPosition();
  LogDebug("") << "[ElectronSeedGenerator::prepareElTrackSeed] outer PixelHit   x,y,z " << outerhit->globalPosition();

  recHits_.clear();

  SiPixelRecHit *pixhit = nullptr;
  SiStripMatchedRecHit2D *striphit = nullptr;
  const SiPixelRecHit *constpixhit = dynamic_cast<const SiPixelRecHit *>(innerhit->hit());
  if (constpixhit) {
    pixhit = new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit);
  } else
    return false;
  constpixhit = dynamic_cast<const SiPixelRecHit *>(outerhit->hit());
  if (constpixhit) {
    pixhit = new SiPixelRecHit(*constpixhit);
    recHits_.push_back(pixhit);
  } else {
    const SiStripMatchedRecHit2D *conststriphit = dynamic_cast<const SiStripMatchedRecHit2D *>(outerhit->hit());
    if (conststriphit) {
      striphit = new SiStripMatchedRecHit2D(*conststriphit);
      recHits_.push_back(striphit);
    } else
      return false;
  }

  // FIXME to be optimized outside the loop
  edm::ESHandle<MagneticField> bfield;
  setup_->get<IdealMagneticFieldRecord>().get(bfield);
  float nomField = bfield->nominalValue();

  // make a spiral
  FastHelix helix(outerhit->globalPosition(), innerhit->globalPosition(), vertexPos, nomField, &*bfield);
  if (!helix.isValid()) {
    return false;
  }
  FreeTrajectoryState fts(helix.stateAtVertex());
  auto propagatedState = propagator_->propagate(fts, innerhit->det()->surface());
  if (!propagatedState.isValid())
    return false;
  auto updatedState = updator_.update(propagatedState, *innerhit);

  auto propagatedState_out = propagator_->propagate(updatedState, outerhit->det()->surface());
  if (!propagatedState_out.isValid())
    return false;
  auto updatedState_out = updator_.update(propagatedState_out, *outerhit);
  // debug prints
  LogDebug("") << "[ElectronSeedGenerator::prepareElTrackSeed] final TSOS, position: "
               << updatedState_out.globalPosition() << " momentum: " << updatedState_out.globalMomentum();
  LogDebug("") << "[ElectronSeedGenerator::prepareElTrackSeed] final TSOS Pt: "
               << updatedState_out.globalMomentum().perp();
  pts_ = trajectoryStateTransform::persistentState(updatedState_out, outerhit->geographicalId().rawId());

  return true;
}
