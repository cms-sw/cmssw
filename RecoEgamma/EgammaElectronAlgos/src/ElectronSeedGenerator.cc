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
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <utility>

namespace {

  bool equivalent(const TrajectorySeed &s1, const TrajectorySeed &s2) {
    if (s1.nHits() != s2.nHits())
      return false;

    const TrajectorySeed::RecHitRange r1 = s1.recHits();
    const TrajectorySeed::RecHitRange r2 = s2.recHits();
    for (auto i1 = r1.begin(), i2 = r2.begin(); i1 != r1.end(); ++i1, ++i2) {
      if (!i1->isValid() || !i2->isValid())
        return false;
      if (i1->geographicalId() != i2->geographicalId())
        return false;
      if (!(i1->localPosition() == i2->localPosition()))
        return false;
    }

    return true;
  }

  void addSeed(reco::ElectronSeed &seed, const SeedWithInfo *info, bool positron, reco::ElectronSeedCollection &out) {
    if (!info) {
      out.emplace_back(seed);
      return;
    }

    if (positron) {
      seed.setPosAttributes(info->dRz2, info->dPhi2, info->dRz1, info->dPhi1);
    } else {
      seed.setNegAttributes(info->dRz2, info->dPhi2, info->dRz1, info->dPhi1);
    }
    for (auto &res : out) {
      if ((seed.caloCluster().key() == res.caloCluster().key()) && (seed.hitsMask() == res.hitsMask()) &&
          equivalent(seed, res)) {
        if (positron) {
          if (res.dRZPos(1) == std::numeric_limits<float>::infinity() &&
              res.dRZNeg(1) != std::numeric_limits<float>::infinity()) {
            res.setPosAttributes(info->dRz2, info->dPhi2, info->dRz1, info->dPhi1);
            seed.setNegAttributes(res.dRZNeg(1), res.dPhiNeg(1), res.dRZNeg(0), res.dPhiNeg(0));
            break;
          } else {
            if (res.dRZPos(1) != std::numeric_limits<float>::infinity()) {
              if (res.dRZPos(1) != seed.dRZPos(1)) {
                edm::LogWarning("ElectronSeedGenerator|BadValue")
                    << "this similar old seed already has another dRz2Pos"
                    << "\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)res.hitsMask() << "/"
                    << res.dRZNeg(1) << "/" << res.dPhiNeg(1) << "/" << res.dRZPos(1) << "/" << res.dPhiPos(1)
                    << "\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)seed.hitsMask() << "/"
                    << seed.dRZNeg(1) << "/" << seed.dPhiNeg(1) << "/" << seed.dRZPos(1) << "/" << seed.dPhiPos(1);
              }
            }
          }
        } else {
          if (res.dRZNeg(1) == std::numeric_limits<float>::infinity() &&
              res.dRZPos(1) != std::numeric_limits<float>::infinity()) {
            res.setNegAttributes(info->dRz2, info->dPhi2, info->dRz1, info->dPhi1);
            seed.setPosAttributes(res.dRZPos(1), res.dPhiPos(1), res.dRZPos(0), res.dPhiPos(0));
            break;
          } else {
            if (res.dRZNeg(1) != std::numeric_limits<float>::infinity()) {
              if (res.dRZNeg(1) != seed.dRZNeg(1)) {
                edm::LogWarning("ElectronSeedGenerator|BadValue")
                    << "this old seed already has another dRz2"
                    << "\nold seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)res.hitsMask() << "/"
                    << res.dRZNeg(1) << "/" << res.dPhiNeg(1) << "/" << res.dRZPos(1) << "/" << res.dPhiPos(1)
                    << "\nnew seed mask/dRz2/dPhi2/dRz2Pos/dPhi2Pos: " << (unsigned int)seed.hitsMask() << "/"
                    << seed.dRZNeg(1) << "/" << seed.dPhiNeg(1) << "/" << seed.dRZPos(1) << "/" << seed.dPhiPos(1);
              }
            }
          }
        }
      }
    }

    out.emplace_back(seed);
  }

  void seedsFromTrajectorySeeds(const std::vector<SeedWithInfo> &pixelSeeds,
                                const reco::ElectronSeed::CaloClusterRef &cluster,
                                reco::ElectronSeedCollection &out,
                                bool positron) {
    if (!pixelSeeds.empty()) {
      LogDebug("ElectronSeedGenerator") << "Compatible " << (positron ? "positron" : "electron") << " seeds found.";
    }

    std::vector<SeedWithInfo>::const_iterator s;
    for (s = pixelSeeds.begin(); s != pixelSeeds.end(); s++) {
      reco::ElectronSeed seed(s->seed);
      seed.setCaloCluster(cluster);
      seed.initTwoHitSeed(s->hitsMask);
      addSeed(seed, &*s, positron, out);
    }
  }

}  // namespace

ElectronSeedGenerator::ElectronSeedGenerator(const edm::ParameterSet &pset,
                                             const ElectronSeedGenerator::Tokens &ts,
                                             edm::ConsumesCollector &&cc)
    : dynamicPhiRoad_(pset.getParameter<bool>("dynamicPhiRoad")),
      verticesTag_(ts.token_vtx),
      beamSpotTag_(ts.token_bs),
      magFieldToken_{cc.esConsumes()},
      trackerGeometryToken_{cc.esConsumes()},
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
      // use of reco vertex
      useRecoVertex_(pset.getParameter<bool>("useRecoVertex")),
      // new B/F configurables
      deltaPhi2B_(pset.getParameter<double>("DeltaPhi2B")),
      deltaPhi2F_(pset.getParameter<double>("DeltaPhi2F")),
      phiMin2B_(-pset.getParameter<double>("PhiMax2B")),
      phiMin2F_(-pset.getParameter<double>("PhiMax2F")),
      phiMax2B_(pset.getParameter<double>("PhiMax2B")),
      phiMax2F_(pset.getParameter<double>("PhiMax2F")),
      matcher_(pset.getParameter<double>("ePhiMin1"),
               pset.getParameter<double>("ePhiMax1"),
               phiMin2B_,
               phiMax2B_,
               phiMin2F_,
               phiMax2F_,
               pset.getParameter<double>("z2MaxB"),
               pset.getParameter<double>("r2MaxF"),
               pset.getParameter<double>("rMaxI"),
               useRecoVertex_) {}

void ElectronSeedGenerator::setupES(const edm::EventSetup &setup) {
  auto newMagField = magneticFieldWatcher_.check(setup);
  auto newTrackerGeom = trackerGeometryWatcher_.check(setup);
  if (newMagField || newTrackerGeom) {
    matcher_.setES(setup.getData(magFieldToken_), setup.getData(trackerGeometryToken_));
  }
}

void ElectronSeedGenerator::run(edm::Event &e,
                                const reco::SuperClusterRefVector &sclRefs,
                                const std::vector<const TrajectorySeedCollection *> &seedsV,
                                reco::ElectronSeedCollection &out) {
  initialSeedCollectionVector_ = &seedsV;

  // get the beamspot from the Event:
  auto const &beamSpot = e.get(beamSpotTag_);

  // if required get the vertices
  std::vector<reco::Vertex> const *vertices = nullptr;
  if (useRecoVertex_)
    vertices = &e.get(verticesTag_);

  for (unsigned int i = 0; i < sclRefs.size(); ++i) {
    // Find the seeds
    LogDebug("ElectronSeedGenerator") << "new cluster, calling seedsFromThisCluster";
    seedsFromThisCluster(sclRefs[i], beamSpot, vertices, out);
  }

  LogDebug("ElectronSeedGenerator") << ": For event " << e.id();
  LogDebug("ElectronSeedGenerator") << "Nr of superclusters after filter: " << sclRefs.size()
                                    << ", no. of ElectronSeeds found  = " << out.size();
}

void ElectronSeedGenerator::seedsFromThisCluster(edm::Ref<reco::SuperClusterCollection> seedCluster,
                                                 reco::BeamSpot const &beamSpot,
                                                 std::vector<reco::Vertex> const *vertices,
                                                 reco::ElectronSeedCollection &out) {
  float clusterEnergy = seedCluster->energy();
  GlobalPoint clusterPos(seedCluster->position().x(), seedCluster->position().y(), seedCluster->position().z());
  reco::ElectronSeed::CaloClusterRef caloCluster(seedCluster);

  if (dynamicPhiRoad_) {
    float clusterEnergyT = clusterEnergy / cosh(EleRelPoint(clusterPos, beamSpot.position()).eta());

    float deltaPhi1;
    if (clusterEnergyT < lowPtThresh_) {
      deltaPhi1 = deltaPhi1Low_;
    } else if (clusterEnergyT > highPtThresh_) {
      deltaPhi1 = deltaPhi1High_;
    } else {
      deltaPhi1 = dPhi1Coef1_ + dPhi1Coef2_ / clusterEnergyT;
    }

    matcher_.set1stLayer(-deltaPhi1 * sizeWindowENeg_, deltaPhi1 * (1. - sizeWindowENeg_));
    matcher_.set2ndLayer(-deltaPhi2B_ / 2., deltaPhi2B_ / 2., -deltaPhi2F_ / 2., deltaPhi2F_ / 2.);
  }

  if (!useRecoVertex_)  // here use the beam spot position
  {
    double sigmaZ = beamSpot.sigmaZ();
    double sigmaZ0Error = beamSpot.sigmaZ0Error();
    double sq = sqrt(sigmaZ * sigmaZ + sigmaZ0Error * sigmaZ0Error);
    double myZmin1 = beamSpot.position().z() - nSigmasDeltaZ1_ * sq;
    double myZmax1 = beamSpot.position().z() + nSigmasDeltaZ1_ * sq;

    GlobalPoint vertexPos;
    ele_convert(beamSpot.position(), vertexPos);

    matcher_.set1stLayerZRange(myZmin1, myZmax1);

    // try electron
    auto elePixelSeeds = matcher_(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, -1.);
    seedsFromTrajectorySeeds(elePixelSeeds, caloCluster, out, false);
    // try positron
    auto posPixelSeeds = matcher_(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, 1.);
    seedsFromTrajectorySeeds(posPixelSeeds, caloCluster, out, true);

  } else if (vertices)  // here we use the reco vertices
  {
    for (auto const &vertex : *vertices) {
      GlobalPoint vertexPos(vertex.position().x(), vertex.position().y(), vertex.position().z());
      double myZmin1, myZmax1;
      if (vertexPos.z() == beamSpot.position().z()) {  // in case vetex not found
        double sigmaZ = beamSpot.sigmaZ();
        double sigmaZ0Error = beamSpot.sigmaZ0Error();
        double sq = sqrt(sigmaZ * sigmaZ + sigmaZ0Error * sigmaZ0Error);
        myZmin1 = beamSpot.position().z() - nSigmasDeltaZ1_ * sq;
        myZmax1 = beamSpot.position().z() + nSigmasDeltaZ1_ * sq;
      } else {  // a vertex has been recoed
        myZmin1 = vertex.position().z() - deltaZ1WithVertex_;
        myZmax1 = vertex.position().z() + deltaZ1WithVertex_;
      }

      matcher_.set1stLayerZRange(myZmin1, myZmax1);

      // try electron
      auto elePixelSeeds = matcher_(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, -1.);
      seedsFromTrajectorySeeds(elePixelSeeds, caloCluster, out, false);
      // try positron
      auto posPixelSeeds = matcher_(*initialSeedCollectionVector_, clusterPos, vertexPos, clusterEnergy, 1.);
      seedsFromTrajectorySeeds(posPixelSeeds, caloCluster, out, true);
    }
  }
}
