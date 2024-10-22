/**
 * \file GE0SegAlgRU.cc
 *
 *  \author I. J. Watson adaptations for GE0
 *  \author M. Maggi for ME0
 *  \from V.Palichik & N.Voytishin
 *  \some functions and structure taken from SK algo by M.Sani and SegFit class by T.Cox
 */
#include "GE0SegAlgoRU.h"
#include "MuonSegFit.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

GE0SegAlgoRU::GE0SegAlgoRU(const edm::ParameterSet& ps) : GEMSegmentAlgorithmBase(ps), myName("GE0SegAlgoRU") {
  allowWideSegments = ps.getParameter<bool>("allowWideSegments");
  doCollisions = ps.getParameter<bool>("doCollisions");

  stdParameters.maxChi2Additional = ps.getParameter<double>("maxChi2Additional");
  stdParameters.maxChi2Prune = ps.getParameter<double>("maxChi2Prune");
  stdParameters.maxChi2GoodSeg = ps.getParameter<double>("maxChi2GoodSeg");
  stdParameters.maxPhiSeeds = ps.getParameter<double>("maxPhiSeeds");
  stdParameters.maxPhiAdditional = ps.getParameter<double>("maxPhiAdditional");
  stdParameters.maxETASeeds = ps.getParameter<double>("maxETASeeds");
  stdParameters.requireCentralBX = ps.getParameter<bool>("requireCentralBX");
  stdParameters.minNumberOfHits = ps.getParameter<unsigned int>("minNumberOfHits");
  stdParameters.maxNumberOfHits = ps.getParameter<unsigned int>("maxNumberOfHits");
  stdParameters.maxNumberOfHitsPerLayer = ps.getParameter<unsigned int>("maxNumberOfHitsPerLayer");
  stdParameters.requireBeamConstr = true;

  wideParameters = stdParameters;
  wideParameters.maxChi2Prune *= 2;
  wideParameters.maxChi2GoodSeg *= 2;
  wideParameters.maxPhiSeeds *= 2;
  wideParameters.maxPhiAdditional *= 2;
  wideParameters.minNumberOfHits += 1;

  displacedParameters = stdParameters;
  displacedParameters.maxChi2Additional *= 2;
  displacedParameters.maxChi2Prune = 100;
  displacedParameters.maxChi2GoodSeg *= 5;
  displacedParameters.maxPhiSeeds *= 2;
  displacedParameters.maxPhiAdditional *= 2;
  displacedParameters.maxETASeeds *= 2;
  displacedParameters.requireBeamConstr = false;

  LogDebug("GE0SegAlgoRU") << myName << " has algorithm cuts set to: \n"
                           << "--------------------------------------------------------------------\n"
                           << "allowWideSegments   = " << allowWideSegments << "\n"
                           << "doCollisions        = " << doCollisions << "\n"
                           << "maxChi2Additional   = " << stdParameters.maxChi2Additional << "\n"
                           << "maxChi2Prune        = " << stdParameters.maxChi2Prune << "\n"
                           << "maxChi2GoodSeg      = " << stdParameters.maxChi2GoodSeg << "\n"
                           << "maxPhiSeeds         = " << stdParameters.maxPhiSeeds << "\n"
                           << "maxPhiAdditional    = " << stdParameters.maxPhiAdditional << "\n"
                           << "maxETASeeds         = " << stdParameters.maxETASeeds << "\n"
                           << std::endl;

  theChamber = nullptr;
}

std::vector<GEMSegment> GE0SegAlgoRU::run(const GEMEnsemble& ensemble, const std::vector<const GEMRecHit*>& rechits) {
  HitAndPositionContainer hitAndPositions;
  auto& superchamber = ensemble.first;
  for (const auto& rechit : rechits) {
    const GEMEtaPartition* part = ensemble.second.at(rechit->gemId().rawId());
    GlobalPoint glb = part->toGlobal(rechit->localPosition());
    LocalPoint nLoc = superchamber->toLocal(glb);
    hitAndPositions.emplace_back(&(*rechit), nLoc, glb, hitAndPositions.size());
  }

  LogDebug("GE0Segment|GE0") << "found " << hitAndPositions.size() << " rechits in superchamber " << superchamber->id();
  //sort by layer
  float z1 = superchamber->chamber(1)->position().z();
  float zlast = superchamber->chamber(superchamber->nChambers())->position().z();
  if (z1 < zlast)
    std::sort(hitAndPositions.begin(), hitAndPositions.end(), [](const HitAndPosition& h1, const HitAndPosition& h2) {
      return h1.layer < h2.layer;
    });
  else
    std::sort(hitAndPositions.begin(), hitAndPositions.end(), [](const HitAndPosition& h1, const HitAndPosition& h2) {
      return h1.layer > h2.layer;
    });
  return run(ensemble.first, hitAndPositions);
}

std::vector<GEMSegment> GE0SegAlgoRU::run(const GEMSuperChamber* chamber, const HitAndPositionContainer& rechits) {
#ifdef EDM_ML_DEBUG  // have lines below only compiled when in debug mode
  GEMDetId chId(chamber->id());
  edm::LogVerbatim("GE0SegAlgoRU") << "[GEMSegmentAlgorithm::run] build segments in chamber " << chId
                                   << " which contains " << rechits.size() << " rechits";
  for (const auto& h : rechits) {
    auto ge0id = h.rh->gemId();
    auto rhLP = h.lp;
    edm::LogVerbatim("GE0SegAlgoRU") << "[RecHit :: Loc x = " << std::showpos << std::setw(9) << rhLP.x()
                                     << " Glb y = " << std::showpos << std::setw(9) << rhLP.y()
                                     << " Time = " << std::showpos << h.rh->BunchX() << " -- " << ge0id.rawId() << " = "
                                     << ge0id << " ]" << std::endl;
  }
#endif

  if (rechits.size() < stdParameters.minNumberOfHits || rechits.size() > stdParameters.maxNumberOfHits) {
    return std::vector<GEMSegment>();
  }

  theChamber = chamber;

  std::vector<unsigned int> recHits_per_layer(theChamber->nChambers(), 0);
  for (const auto& rechit : rechits) {
    recHits_per_layer[rechit.layer - 1]++;
  }

  BoolContainer used(rechits.size(), false);

  // We have at least 2 hits. We intend to try all possible pairs of hits to start
  // segment building. 'All possible' means each hit lies on different layers in the chamber.
  // after all same size segs are build we get rid of the overcrossed segments using the chi2 criteria
  // the hits from the segs that are left are marked as used and are not added to segs in future iterations
  // the hits from 3p segs are marked as used separately in order to try to assamble them in longer segments
  // in case there is a second pass

  // Choose first hit (as close to IP as possible) h1 and second hit
  // (as far from IP as possible) h2 To do this we iterate over hits
  // in the chamber by layer - pick two layers.  Then we
  // iterate over hits within each of these layers and pick h1 and h2
  // these.  If they are 'close enough' we build an empty
  // segment.  Then try adding hits to this segment.

  std::vector<GEMSegment> segments;

  auto doStd = [&]() {
    for (unsigned int n_seg_min = 6u; n_seg_min >= stdParameters.minNumberOfHits; --n_seg_min)
      lookForSegments(stdParameters, n_seg_min, rechits, recHits_per_layer, used, segments);
  };
  auto doDisplaced = [&]() {
    for (unsigned int n_seg_min = 6u; n_seg_min >= displacedParameters.minNumberOfHits; --n_seg_min)
      lookForSegments(displacedParameters, n_seg_min, rechits, recHits_per_layer, used, segments);
  };
  // Not currently used
  // auto doWide = [&] () {
  // 	for(unsigned int n_seg_min = 6u; n_seg_min >= wideParameters.minNumberOfHits; --n_seg_min)
  // 		lookForSegments(wideParameters,n_seg_min,rechits,recHits_per_layer, used,segments);
  // };
  auto printSegments = [&] {
#ifdef EDM_ML_DEBUG  // have lines below only compiled when in debug mode
    for (const auto& seg : segments) {
      GEMDetId chId(seg.gemDetId());
      const auto& rechits = seg.specificRecHits();
      edm::LogVerbatim("GE0SegAlgoRU") << "[GE0SegAlgoRU] segment in chamber " << chId << " which contains "
                                       << rechits.size() << " rechits and with specs: \n"
                                       << seg;
      for (const auto& rh : rechits) {
        auto ge0id = rh.gemId();
        edm::LogVerbatim("GE0SegAlgoRU") << "[RecHit :: Loc x = " << std::showpos << std::setw(9)
                                         << rh.localPosition().x() << " Loc y = " << std::showpos << std::setw(9)
                                         << rh.localPosition().y() << " Time = " << std::showpos << rh.BunchX()
                                         << " -- " << ge0id.rawId() << " = " << ge0id << " ]";
      }
    }
#endif
  };

  //If we arent doing collisions, do a single iteration
  if (!doCollisions) {
    doDisplaced();
    printSegments();
    return segments;
  }

  //Iteration 1: STD processing
  doStd();

  if (false) {
    //How the CSC algorithm ~does iterations. for now not considering
    //displaced muons will not worry about it  later
    //Iteration 2a: If we don't allow wide segments simply do displaced
    // Or if we already found a segment simply skip to displaced
    if (!allowWideSegments || !segments.empty()) {
      doDisplaced();
      return segments;
    }
    //doWide();
    doDisplaced();
  }
  printSegments();
  return segments;
}

void GE0SegAlgoRU::lookForSegments(const SegmentParameters& params,
                                   const unsigned int n_seg_min,
                                   const HitAndPositionContainer& rechits,
                                   const std::vector<unsigned int>& recHits_per_layer,
                                   BoolContainer& used,
                                   std::vector<GEMSegment>& segments) const {
  auto ib = rechits.begin();
  auto ie = rechits.end();
  std::vector<std::pair<float, HitAndPositionPtrContainer> > proto_segments;
  // the first hit is taken from the back
  for (auto i1 = ib; i1 != ie; ++i1) {
    const auto& h1 = *i1;

    //skip if rh is used and the layer tat has big rh multiplicity(>25RHs)
    if (used[h1.idx])
      continue;
    if (recHits_per_layer[h1.layer - 1] > params.maxNumberOfHitsPerLayer)
      continue;

    // the second hit from the front
    for (auto i2 = ie - 1; i2 != i1; --i2) {
      const auto& h2 = *i2;

      //skip if rh is used and the layer tat has big rh multiplicity(>25RHs)
      if (used[h2.idx])
        continue;
      if (recHits_per_layer[h2.layer - 1] > params.maxNumberOfHitsPerLayer)
        continue;

      //Stop if the distance between layers is not large enough
      if ((std::abs(int(h2.layer) - int(h1.layer)) + 1) < int(n_seg_min))
        break;

      if (!areHitsCloseInEta(params.maxETASeeds, params.requireBeamConstr, h1.gp, h2.gp))
        continue;
      if (!areHitsCloseInGlobalPhi(params.maxPhiSeeds, std::abs(int(h2.layer) - int(h1.layer)), h1.gp, h2.gp))
        continue;

      HitAndPositionPtrContainer current_proto_segment;
      std::unique_ptr<MuonSegFit> current_fit;
      current_fit = addHit(current_proto_segment, h1);
      current_fit = addHit(current_proto_segment, h2);

      tryAddingHitsToSegment(params.maxETASeeds,
                             params.maxPhiAdditional,
                             params.maxChi2Additional,
                             current_fit,
                             current_proto_segment,
                             used,
                             i1,
                             i2);

      if (current_proto_segment.size() > n_seg_min)
        pruneBadHits(params.maxChi2Prune, current_proto_segment, current_fit, n_seg_min);

      LogDebug("GE0SegAlgoRU") << "[GE0SegAlgoRU::lookForSegments] # of hits in segment "
                               << current_proto_segment.size() << " min # " << n_seg_min << " => "
                               << (current_proto_segment.size() >= n_seg_min) << " chi2/ndof "
                               << current_fit->chi2() / current_fit->ndof() << " => "
                               << (current_fit->chi2() / current_fit->ndof() < params.maxChi2GoodSeg) << std::endl;

      if (current_proto_segment.size() < n_seg_min)
        continue;
      const float current_metric = current_fit->chi2() / current_fit->ndof();
      if (current_metric > params.maxChi2GoodSeg)
        continue;

      if (params.requireCentralBX) {
        int nCentral = 0;
        int nNonCentral = 0;
        for (const auto* rh : current_proto_segment) {
          if (std::abs(rh->rh->BunchX()) < 2)
            nCentral++;
          else
            nNonCentral++;
        }
        if (nNonCentral >= nCentral)
          continue;
      }

      proto_segments.emplace_back(current_metric, current_proto_segment);
    }
  }
  addUniqueSegments(proto_segments, segments, used);
}

void GE0SegAlgoRU::addUniqueSegments(SegmentByMetricContainer& proto_segments,
                                     std::vector<GEMSegment>& segments,
                                     BoolContainer& used) const {
  std::sort(proto_segments.begin(),
            proto_segments.end(),
            [](const std::pair<float, HitAndPositionPtrContainer>& a,
               const std::pair<float, HitAndPositionPtrContainer>& b) { return a.first < b.first; });

  //Now add to the collect based on minChi2 marking the hits as used after
  std::vector<unsigned int> usedHits;
  for (auto& container : proto_segments) {
    HitAndPositionPtrContainer currentProtoSegment = container.second;

    //check to see if we already used thes hits this round
    bool alreadyFilled = false;
    for (const auto& h : currentProtoSegment) {
      for (unsigned int iOH = 0; iOH < usedHits.size(); ++iOH) {
        if (usedHits[iOH] != h->idx)
          continue;
        alreadyFilled = true;
        break;
      }
    }
    if (alreadyFilled)
      continue;
    for (const auto* h : currentProtoSegment) {
      usedHits.push_back(h->idx);
      used[h->idx] = true;
    }

    std::unique_ptr<MuonSegFit> current_fit = makeFit(currentProtoSegment);

    // Create an actual GEMSegment - retrieve all info from the fit
    // calculate the timing fron rec hits associated to the TrackingRecHits used
    // to fit the segment
    float averageBX = 0.;
    for (const auto* h : currentProtoSegment) {
      averageBX += h->rh->BunchX();
    }
    averageBX /= int(currentProtoSegment.size());

    std::sort(currentProtoSegment.begin(),
              currentProtoSegment.end(),
              [](const HitAndPosition* a, const HitAndPosition* b) { return a->layer < b->layer; });

    std::vector<const GEMRecHit*> bareRHs;
    bareRHs.reserve(currentProtoSegment.size());
    for (const auto* rh : currentProtoSegment)
      bareRHs.push_back(rh->rh);
    const float dPhi = theChamber->computeDeltaPhi(current_fit->intercept(), current_fit->localdir());
    GEMSegment temp(bareRHs,
                    current_fit->intercept(),
                    current_fit->localdir(),
                    current_fit->covarianceMatrix(),
                    current_fit->chi2(),
                    averageBX,
                    dPhi);
    segments.push_back(temp);
  }
}

void GE0SegAlgoRU::tryAddingHitsToSegment(const float maxETA,
                                          const float maxPhi,
                                          const float maxChi2,
                                          std::unique_ptr<MuonSegFit>& current_fit,
                                          HitAndPositionPtrContainer& proto_segment,
                                          const BoolContainer& used,
                                          HitAndPositionContainer::const_iterator i1,
                                          HitAndPositionContainer::const_iterator i2) const {
  // Iterate over the layers with hits in the chamber
  // Skip the layers containing the segment endpoints
  // Test each hit on the other layers to see if it is near the segment
  // If it is, see whether there is already a hit on the segment from the same layer
  //    - if so, and there are more than 2 hits on the segment, copy the segment,
  //      replace the old hit with the new hit. If the new segment chi2 is better
  //      then replace the original segment with the new one (by swap)
  //    - if not, copy the segment, add the hit. If the new chi2/dof is still satisfactory
  //      then replace the original segment with the new one (by swap)

  //Hits are ordered by layer, "i1" is the inner hit and i2 is the outer hit
  //so possible hits to add must be between these two iterators
  for (auto iH = i1 + 1; iH != i2; ++iH) {
    if (iH->layer == i1->layer)
      continue;
    if (iH->layer == i2->layer)
      break;
    if (used[iH->idx])
      continue;
    if (!isHitNearSegment(maxETA, maxPhi, current_fit, proto_segment, *iH))
      continue;
    if (hasHitOnLayer(proto_segment, iH->layer))
      compareProtoSegment(current_fit, proto_segment, *iH);
    else
      increaseProtoSegment(maxChi2, current_fit, proto_segment, *iH);
  }
}

bool GE0SegAlgoRU::areHitsCloseInEta(const float maxETA,
                                     const bool beamConst,
                                     const GlobalPoint& h1,
                                     const GlobalPoint& h2) const {
  float diff = std::abs(h1.eta() - h2.eta());
  LogDebug("GE0SegAlgoRU") << "[GE0SegAlgoRU::areHitsCloseInEta] gp1 = " << h1 << " in eta part = " << h1.eta()
                           << " and gp2 = " << h2 << " in eta part = " << h2.eta() << " ==> dEta = " << diff
                           << " ==> return " << (diff < 0.1) << std::endl;
  //temp for floating point comparision...maxEta is the difference between partitions, so x1.5 to take into account non-circle geom.
  return (diff < std::max(maxETA, 0.01f));
}

bool GE0SegAlgoRU::areHitsCloseInGlobalPhi(const float maxPHI,
                                           const unsigned int nLayDisp,
                                           const GlobalPoint& h1,
                                           const GlobalPoint& h2) const {
  float dphi12 = deltaPhi(h1.barePhi(), h2.barePhi());
  LogDebug("GE0SegAlgoRU") << "[GE0SegAlgoRU::areHitsCloseInGlobalPhi] gp1 = " << h1 << " and gp2 = " << h2
                           << " ==> dPhi = " << dphi12 << " ==> return " << (std::abs(dphi12) < std::max(maxPHI, 0.02f))
                           << std::endl;
  return std::abs(dphi12) < std::max(maxPHI, float(float(nLayDisp) * 0.004));
}

bool GE0SegAlgoRU::isHitNearSegment(const float maxETA,
                                    const float maxPHI,
                                    const std::unique_ptr<MuonSegFit>& fit,
                                    const HitAndPositionPtrContainer& proto_segment,
                                    const HitAndPosition& h) const {
  //Get average eta, based on the two seeds...asssumes that we have not started pruning yet!
  const float avgETA = (proto_segment[1]->gp.eta() + proto_segment[0]->gp.eta()) / 2.;
  if (std::abs(h.gp.eta() - avgETA) > std::max(maxETA, 0.01f))
    return false;

  //Now check the dPhi based on the segment fit
  GlobalPoint globIntercept = globalAtZ(fit, h.lp.z());
  float dPhi = deltaPhi(h.gp.barePhi(), globIntercept.phi());
  //check to see if it is inbetween the two rolls of the outer and inner hits
  return (std::abs(dPhi) < std::max(maxPHI, 0.001f));
}

GlobalPoint GE0SegAlgoRU::globalAtZ(const std::unique_ptr<MuonSegFit>& fit, float z) const {
  float x = fit->xfit(z);
  float y = fit->yfit(z);
  return theChamber->toGlobal(LocalPoint(x, y, z));
}

std::unique_ptr<MuonSegFit> GE0SegAlgoRU::addHit(HitAndPositionPtrContainer& proto_segment,
                                                 const HitAndPosition& aHit) const {
  proto_segment.push_back(&aHit);
  // make a fit
  return makeFit(proto_segment);
}

std::unique_ptr<MuonSegFit> GE0SegAlgoRU::makeFit(const HitAndPositionPtrContainer& proto_segment) const {
  // for GE0 we take the gemrechit from the proto_segment we transform into Tracking Rechits
  // the local rest frame is the GEMSuperChamber
  MuonSegFit::MuonRecHitContainer muonRecHits;
  for (const auto& rh : proto_segment) {
    GEMRecHit* newRH = rh->rh->clone();
    newRH->setPosition(rh->lp);
    MuonSegFit::MuonRecHitPtr trkRecHit(newRH);
    muonRecHits.push_back(trkRecHit);
  }
  auto currentFit = std::make_unique<MuonSegFit>(muonRecHits);
  currentFit->fit();
  return currentFit;
}

void GE0SegAlgoRU::pruneBadHits(const float maxChi2,
                                HitAndPositionPtrContainer& proto_segment,
                                std::unique_ptr<MuonSegFit>& fit,
                                const unsigned int n_seg_min) const {
  while (proto_segment.size() > n_seg_min && fit->chi2() / fit->ndof() > maxChi2) {
    float maxDev = -1;
    HitAndPositionPtrContainer::iterator worstHit;
    for (auto it = proto_segment.begin(); it != proto_segment.end(); ++it) {
      const float dev = getHitSegChi2(fit, *(*it)->rh);
      if (dev < maxDev)
        continue;
      maxDev = dev;
      worstHit = it;
    }
    LogDebug("GE0SegAlgoRU") << "[GE0SegAlgoRU::pruneBadHits] pruning one hit-> layer: " << (*worstHit)->layer
                             << " eta: " << (*worstHit)->gp.eta() << " phi: " << (*worstHit)->gp.phi()
                             << " old chi2/dof: " << fit->chi2() / fit->ndof() << std::endl;
    proto_segment.erase(worstHit);
    fit = makeFit(proto_segment);
  }
}

float GE0SegAlgoRU::getHitSegChi2(const std::unique_ptr<MuonSegFit>& fit, const GEMRecHit& hit) const {
  const auto lp = hit.localPosition();
  const auto le = hit.localPositionError();
  const float du = fit->xdev(lp.x(), lp.z());
  const float dv = fit->ydev(lp.y(), lp.z());

  ROOT::Math::SMatrix<double, 2, 2, ROOT::Math::MatRepSym<double, 2> > IC;
  IC(0, 0) = le.xx();
  IC(1, 0) = le.xy();
  IC(1, 1) = le.yy();

  // Invert covariance matrix
  IC.Invert();
  return du * du * IC(0, 0) + 2. * du * dv * IC(0, 1) + dv * dv * IC(1, 1);
}

bool GE0SegAlgoRU::hasHitOnLayer(const HitAndPositionPtrContainer& proto_segment, const unsigned int layer) const {
  for (const auto* h : proto_segment)
    if (h->layer == layer)
      return true;
  return false;
}

void GE0SegAlgoRU::compareProtoSegment(std::unique_ptr<MuonSegFit>& current_fit,
                                       HitAndPositionPtrContainer& current_proto_segment,
                                       const HitAndPosition& new_hit) const {
  const HitAndPosition* old_hit = nullptr;
  HitAndPositionPtrContainer new_proto_segment = current_proto_segment;

  for (auto it = new_proto_segment.begin(); it != new_proto_segment.end();) {
    if ((*it)->layer == new_hit.layer) {
      old_hit = *it;
      it = new_proto_segment.erase(it);
    } else {
      ++it;
    }
  }
  if (old_hit == nullptr)
    return;
  auto new_fit = addHit(new_proto_segment, new_hit);

  //If on the same strip but different BX choose the closest
  bool useNew = false;
  if (old_hit->lp == new_hit.lp) {
    float avgbx = 0;
    for (const auto* h : current_proto_segment)
      if (old_hit != h)
        avgbx += h->rh->BunchX();
    avgbx /= float(current_proto_segment.size() - 1);
    if (std::abs(avgbx - new_hit.rh->BunchX()) < std::abs(avgbx - old_hit->rh->BunchX()))
      useNew = true;
  }  //otherwise base it on chi2
  else if (new_fit->chi2() < current_fit->chi2())
    useNew = true;

  if (useNew) {
    current_proto_segment = new_proto_segment;
    current_fit = std::move(new_fit);
  }
}

void GE0SegAlgoRU::increaseProtoSegment(const float maxChi2,
                                        std::unique_ptr<MuonSegFit>& current_fit,
                                        HitAndPositionPtrContainer& current_proto_segment,
                                        const HitAndPosition& new_hit) const {
  HitAndPositionPtrContainer new_proto_segment = current_proto_segment;
  auto new_fit = addHit(new_proto_segment, new_hit);
  if (new_fit->chi2() / new_fit->ndof() < maxChi2) {
    current_proto_segment = new_proto_segment;
    current_fit = std::move(new_fit);
  }
}
