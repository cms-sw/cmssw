#include "RecoMuon/TrackerSeedGenerator/interface/SeedMvaEstimatorPhase2.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

#include <cmath>

using namespace std;

namespace Phase2 {
  enum inputIndexesPhase2 {
    kTsosErr0,           // 0
    kTsosErr2,           // 1
    kTsosErr5,           // 2
    kTsosDxdz,           // 3
    kTsosDydz,           // 4
    kTsosQbp,            // 5
    kDRdRL1TkMuSeedP,    // 6
    kDRdPhiL1TkMuSeedP,  // 7
    kExpd2HitL1Tk1,      // 8
    kExpd2HitL1Tk2,      // 9
    kExpd2HitL1Tk3,      // 10
    kLast                // 11
  };
}

SeedMvaEstimatorPhase2::SeedMvaEstimatorPhase2(const edm::FileInPath& weightsfile,
                                               const std::vector<double>& scale_mean,
                                               const std::vector<double>& scale_std)
    : scale_mean_(scale_mean), scale_std_(scale_std) {
  gbrForest_ = createGBRForest(weightsfile);
}

SeedMvaEstimatorPhase2::~SeedMvaEstimatorPhase2() {}

void SeedMvaEstimatorPhase2::getL1TTVariables(const TrajectorySeed& seed,
                                              const GlobalVector& global_p,
                                              const GlobalPoint& global_x,
                                              const edm::Handle<l1t::TrackerMuonCollection>& h_L1TkMu,
                                              float& DRL1TkMu,
                                              float& DPhiL1TkMu) const {
  for (auto L1TkMu = h_L1TkMu->begin(); L1TkMu != h_L1TkMu->end(); ++L1TkMu) {
    auto TkRef = L1TkMu->trkPtr();
    float DRL1TkMu_tmp = reco::deltaR(*TkRef, global_p);
    float DPhiL1TkMu_tmp = reco::deltaPhi(TkRef->phi(), global_p.phi());
    if (DRL1TkMu_tmp < DRL1TkMu) {
      DRL1TkMu = DRL1TkMu_tmp;
      DPhiL1TkMu = DPhiL1TkMu_tmp;
    }
  }
}

vector<LayerTSOS> SeedMvaEstimatorPhase2::getTsosOnPixels(const TTTrack<Ref_Phase2TrackerDigi_>& l1tk,
                                                          const edm::ESHandle<MagneticField>& magfieldH,
                                                          const Propagator& propagatorAlong,
                                                          const GeometricSearchTracker& geomTracker) const {
  vector<LayerTSOS> v_tsos = {};

  std::vector<BarrelDetLayer const*> const& bpix = geomTracker.pixelBarrelLayers();
  std::vector<ForwardDetLayer const*> const& nfpix = geomTracker.negPixelForwardLayers();
  std::vector<ForwardDetLayer const*> const& pfpix = geomTracker.posPixelForwardLayers();

  int chargeTk = l1tk.rInv() > 0. ? 1 : -1;
  GlobalPoint gpos = l1tk.POCA();
  GlobalVector gmom = l1tk.momentum();

  FreeTrajectoryState fts = FreeTrajectoryState(gpos, gmom, chargeTk, magfieldH.product());

  for (auto it = bpix.begin(); it != bpix.end(); ++it) {
    TrajectoryStateOnSurface tsos = propagatorAlong.propagate(fts, (**it).specificSurface());
    if (!tsos.isValid())
      continue;

    auto z0 = std::abs(tsos.globalPosition().z() - (**it).specificSurface().position().z());
    auto deltaZ = 0.5f * (**it).surface().bounds().length();
    deltaZ += 0.5f * (**it).surface().bounds().thickness() * std::abs(tsos.globalDirection().z()) /
              tsos.globalDirection().perp();
    bool is_compatible = (z0 < deltaZ);

    if (is_compatible) {
      v_tsos.push_back(make_pair((const DetLayer*)(*it), tsos));
    }
  }
  for (auto it = nfpix.begin(); it != nfpix.end(); ++it) {
    TrajectoryStateOnSurface tsos = propagatorAlong.propagate(fts, (**it).specificSurface());
    if (!tsos.isValid())
      continue;

    auto r2 = tsos.localPosition().perp2();
    float deltaR = 0.5f * (**it).surface().bounds().thickness() * tsos.localDirection().perp() /
                   std::abs(tsos.localDirection().z());
    auto ri2 = std::max((**it).specificSurface().innerRadius() - deltaR, 0.f);
    ri2 *= ri2;
    auto ro2 = (**it).specificSurface().outerRadius() + deltaR;
    ro2 *= ro2;
    bool is_compatible = ((r2 > ri2) && (r2 < ro2));

    if (is_compatible) {
      v_tsos.push_back(make_pair((const DetLayer*)(*it), tsos));
    }
  }
  for (auto it = pfpix.begin(); it != pfpix.end(); ++it) {
    TrajectoryStateOnSurface tsos = propagatorAlong.propagate(fts, (**it).specificSurface());
    if (!tsos.isValid())
      continue;

    auto r2 = tsos.localPosition().perp2();
    float deltaR = 0.5f * (**it).surface().bounds().thickness() * tsos.localDirection().perp() /
                   std::abs(tsos.localDirection().z());
    auto ri2 = std::max((**it).specificSurface().innerRadius() - deltaR, 0.f);
    ri2 *= ri2;
    auto ro2 = (**it).specificSurface().outerRadius() + deltaR;
    ro2 *= ro2;
    bool is_compatible = ((r2 > ri2) && (r2 < ro2));
    if (is_compatible) {
      v_tsos.push_back(make_pair((const DetLayer*)(*it), tsos));
    }
  }
  return v_tsos;
}

// FIX HERE
vector<pair<LayerHit, LayerTSOS> > SeedMvaEstimatorPhase2::getHitTsosPairs(
    const TrajectorySeed& seed,
    const edm::Handle<l1t::TrackerMuonCollection>& L1TkMuonHandle,
    const edm::ESHandle<MagneticField>& magfieldH,
    const Propagator& propagatorAlong,
    const GeometricSearchTracker& geomTracker) const {
  vector<pair<LayerHit, LayerTSOS> > out = {};
  float av_dr_min = 999.;

  // -- loop on L1TkMu
  for (auto L1TkMu = L1TkMuonHandle->begin(); L1TkMu != L1TkMuonHandle->end(); ++L1TkMu) {
    const vector<LayerTSOS> v_tsos = getTsosOnPixels(*L1TkMu->trkPtr(), magfieldH, propagatorAlong, geomTracker);

    // -- loop on recHits
    vector<int> v_tsos_skip(v_tsos.size(), 0);
    vector<pair<LayerHit, LayerTSOS> > hitTsosPair = {};
    int ihit = 0;
    for (const auto& hit : seed.recHits()) {
      // -- look for closest tsos by absolute distance
      int the_tsos = -99999;
      float dr_min = 20.;
      for (auto i = 0U; i < v_tsos.size(); ++i) {
        if (v_tsos_skip.at(i))
          continue;
        float dr = (v_tsos.at(i).second.globalPosition() - hit.globalPosition()).mag();
        if (dr < dr_min) {
          dr_min = dr;
          the_tsos = i;
          v_tsos_skip.at(i) = 1;
        }
      }
      if (the_tsos > -1) {
        const DetLayer* thelayer = geomTracker.idToLayer(hit.geographicalId());
        hitTsosPair.push_back(make_pair(make_pair(thelayer, &hit), v_tsos.at(the_tsos)));
      }
      ihit++;
    }  // loop on recHits

    // -- find tsos for all recHits?
    if ((int)hitTsosPair.size() == ihit) {
      float av_dr = 0.;
      for (auto it = hitTsosPair.begin(); it != hitTsosPair.end(); ++it) {
        auto hit = it->first.second;
        auto tsos = it->second.second;
        av_dr += (hit->globalPosition() - tsos.globalPosition()).mag();
      }
      av_dr = av_dr > 0 ? av_dr / (float)hitTsosPair.size() : 1.e6;
      if (av_dr < av_dr_min) {
        av_dr_min = av_dr;
        out = hitTsosPair;
      }
    }
  }  // loop on L1TkMu
  return out;
}

void SeedMvaEstimatorPhase2::getHitL1TkVariables(const TrajectorySeed& seed,
                                                 const edm::Handle<l1t::TrackerMuonCollection>& L1TkMuonHandle,
                                                 const edm::ESHandle<MagneticField>& magfieldH,
                                                 const Propagator& propagatorAlong,
                                                 const GeometricSearchTracker& geomTracker,
                                                 float& expd2HitL1Tk1,
                                                 float& expd2HitL1Tk2,
                                                 float& expd2HitL1Tk3) const {
  vector<pair<LayerHit, LayerTSOS> > hitTsosPair =
      getHitTsosPairs(seed, L1TkMuonHandle, magfieldH, propagatorAlong, geomTracker);

  bool found = (!hitTsosPair.empty());

  if (found) {
    float l1x1 = 99999.;
    float l1y1 = 99999.;
    float l1z1 = 99999.;
    float hitx1 = 99999.;
    float hity1 = 99999.;
    float hitz1 = 99999.;

    float l1x2 = 99999.;
    float l1y2 = 99999.;
    float l1z2 = 99999.;
    float hitx2 = 99999.;
    float hity2 = 99999.;
    float hitz2 = 99999.;

    float l1x3 = 99999.;
    float l1y3 = 99999.;
    float l1z3 = 99999.;
    float hitx3 = 99999.;
    float hity3 = 99999.;
    float hitz3 = 99999.;

    if (hitTsosPair.size() > 1) {
      auto hit1 = hitTsosPair.at(0).first.second;
      auto tsos1 = hitTsosPair.at(0).second.second;

      l1x1 = tsos1.globalPosition().x();
      l1y1 = tsos1.globalPosition().y();
      l1z1 = tsos1.globalPosition().z();
      hitx1 = hit1->globalPosition().x();
      hity1 = hit1->globalPosition().y();
      hitz1 = hit1->globalPosition().z();

      auto hit2 = hitTsosPair.at(1).first.second;
      auto tsos2 = hitTsosPair.at(1).second.second;

      l1x2 = tsos2.globalPosition().x();
      l1y2 = tsos2.globalPosition().y();
      l1z2 = tsos2.globalPosition().z();
      hitx2 = hit2->globalPosition().x();
      hity2 = hit2->globalPosition().y();
      hitz2 = hit2->globalPosition().z();
    }
    if (hitTsosPair.size() > 2) {
      auto hit3 = hitTsosPair.at(2).first.second;
      auto tsos3 = hitTsosPair.at(2).second.second;

      l1x3 = tsos3.globalPosition().x();
      l1y3 = tsos3.globalPosition().y();
      l1z3 = tsos3.globalPosition().z();
      hitx3 = hit3->globalPosition().x();
      hity3 = hit3->globalPosition().y();
      hitz3 = hit3->globalPosition().z();
    }

    // If hit == 99999. >> cannot find hit info >> distance btw hit~tsos is large >> expd2HitL1Tk ~ 0
    if (hitx1 != 99999.) {
      expd2HitL1Tk1 = exp(-(pow((l1x1 - hitx1), 2) + pow((l1y1 - hity1), 2) + pow((l1z1 - hitz1), 2)));
    } else {
      expd2HitL1Tk1 = 0.;
    }

    if (hitx2 != 99999.) {
      expd2HitL1Tk2 = exp(-(pow((l1x2 - hitx2), 2) + pow((l1y2 - hity2), 2) + pow((l1z2 - hitz2), 2)));
    } else {
      expd2HitL1Tk2 = 0.;
    }

    if (hitx3 != 99999.) {
      expd2HitL1Tk3 = exp(-(pow((l1x3 - hitx3), 2) + pow((l1y3 - hity3), 2) + pow((l1z3 - hitz3), 2)));
    } else {
      expd2HitL1Tk3 = 0.;
    }
  }
}

double SeedMvaEstimatorPhase2::computeMva(const TrajectorySeed& seed,
                                          const GlobalVector& global_p,
                                          const GlobalPoint& global_x,
                                          const edm::Handle<l1t::TrackerMuonCollection>& h_L1TkMu,
                                          const edm::ESHandle<MagneticField>& magfieldH,
                                          const Propagator& propagatorAlong,
                                          const GeometricSearchTracker& geomTracker) const {
  float Phase2var[Phase2::kLast]{};

  Phase2var[Phase2::kTsosErr0] = seed.startingState().error(0);
  Phase2var[Phase2::kTsosErr2] = seed.startingState().error(2);
  Phase2var[Phase2::kTsosErr5] = seed.startingState().error(5);
  Phase2var[Phase2::kTsosDxdz] = seed.startingState().parameters().dxdz();
  Phase2var[Phase2::kTsosDydz] = seed.startingState().parameters().dydz();
  Phase2var[Phase2::kTsosQbp] = seed.startingState().parameters().qbp();

  float initDRdPhi = 99999.;

  float DRL1TkMuSeedP = initDRdPhi;
  float DPhiL1TkMuSeedP = initDRdPhi;
  getL1TTVariables(seed, global_p, global_x, h_L1TkMu, DRL1TkMuSeedP, DPhiL1TkMuSeedP);
  Phase2var[Phase2::kDRdRL1TkMuSeedP] = DRL1TkMuSeedP;
  Phase2var[Phase2::kDRdPhiL1TkMuSeedP] = DPhiL1TkMuSeedP;

  float expd2HitL1Tk1 = initDRdPhi;
  float expd2HitL1Tk2 = initDRdPhi;
  float expd2HitL1Tk3 = initDRdPhi;
  getHitL1TkVariables(
      seed, h_L1TkMu, magfieldH, propagatorAlong, geomTracker, expd2HitL1Tk1, expd2HitL1Tk2, expd2HitL1Tk3);
  Phase2var[Phase2::kExpd2HitL1Tk1] = expd2HitL1Tk1;
  Phase2var[Phase2::kExpd2HitL1Tk2] = expd2HitL1Tk2;
  Phase2var[Phase2::kExpd2HitL1Tk3] = expd2HitL1Tk3;

  for (int iv = 0; iv < Phase2::kLast; ++iv) {
    Phase2var[iv] = (Phase2var[iv] - scale_mean_.at(iv)) / scale_std_.at(iv);
  }

  return gbrForest_->GetResponse(Phase2var);
}
