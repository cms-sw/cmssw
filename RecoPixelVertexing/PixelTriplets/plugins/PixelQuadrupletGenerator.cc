#include "PixelQuadrupletGenerator.h"
#include "ThirdHitRZPrediction.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

namespace {
  template <typename T>
  T sqr(T x) {
    return x*x;
  }
}

typedef PixelRecoRange<float> Range;

PixelQuadrupletGenerator::PixelQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")), //extra window in ThirdHitRZPrediction range
  extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
  extraPhiTolerance(cfg.getParameter<edm::ParameterSet>("extraPhiTolerance")),
  maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
  fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
  fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
  useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection"))
{
  if(cfg.exists("SeedComparitorPSet")) {
    edm::ParameterSet comparitorPSet =
      cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
    std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
    if(comparitorName != "none") {
      theComparitor.reset(SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));
    }
  }
}

PixelQuadrupletGenerator::~PixelQuadrupletGenerator() {}


void PixelQuadrupletGenerator::hitQuadruplets(const TrackingRegion& region, OrderedHitSeeds& result,
                                              const edm::Event& ev, const edm::EventSetup& es,
                                              const SeedingLayerSetsHits::SeedingLayerSet& tripletLayers,
                                              const std::vector<SeedingLayerSetsHits::SeedingLayer>& fourthLayers)
{
  if (theComparitor) theComparitor->init(ev, es);

  OrderedHitTriplets triplets;
  theTripletGenerator->hitTriplets(region, triplets, ev, es,
                                   tripletLayers, // pair generator picks the correct two layers from these
                                   std::vector<SeedingLayerSetsHits::SeedingLayer>{tripletLayers[2]});
  if(triplets.empty()) return;

  const size_t size = fourthLayers.size();

  const RecHitsSortedInPhi *fourthHitMap[size];
  typedef RecHitsSortedInPhi::Hit Hit;

  using NodeInfo = KDTreeNodeInfo<unsigned int>;
  std::vector<NodeInfo > layerTree; // re-used throughout
  std::vector<unsigned int> foundNodes; // re-used thoughout

  declareDynArray(KDTreeLinkerAlgo<unsigned int>, size, hitTree);
  float rzError[size]; //save maximum errors

  declareDynArray(ThirdHitRZPrediction<PixelRecoLineRZ>, size, preds);

  // Build KDtrees
  for(size_t il=0; il!=size; ++il) {
    fourthHitMap[il] = &(*theLayerCache)(fourthLayers[il], region, ev, es);
    auto const& hits = *fourthHitMap[il];

    ThirdHitRZPrediction<PixelRecoLineRZ> & pred = preds[il];
    pred.initLayer(fourthLayers[il].detLayer());
    pred.initTolerance(extraHitRZtolerance);

    layerTree.clear();
    float maxphi = Geom::ftwoPi(), minphi = -maxphi; // increase to cater for any range
    float minv=999999.0, maxv= -999999.0; // Initialise to extreme values in case no hits
    float maxErr=0.0f;
    for (unsigned int i=0; i!=hits.size(); ++i) {
      auto angle = hits.phi(i);
      auto v =  hits.gv(i);
      //use (phi,r) for endcaps rather than (phi,z)
      minv = std::min(minv,v);  maxv = std::max(maxv,v);
      float myerr = hits.dv[i];
      maxErr = std::max(maxErr,myerr);
      layerTree.emplace_back(i, angle, v); // save it
      if (angle < 0)  // wrap all points in phi
	{ layerTree.emplace_back(i, angle+Geom::ftwoPi(), v);}
      else
	{ layerTree.emplace_back(i, angle-Geom::ftwoPi(), v);}
    }
    KDTreeBox phiZ(minphi, maxphi, minv-0.01f, maxv+0.01f);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ); // make KDtree
    rzError[il] = maxErr; // save error
  }

  const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);
  const QuantityDependsPtEval extraPhiToleranceEval = extraPhiTolerance.evaluator(es);

  // re-used thoughout
  std::array<float, 4> bc_r;
  std::array<float, 4> bc_z;
  std::array<float, 4> bc_errZ2;
  std::array<GlobalPoint, 4> gps;
  std::array<GlobalError, 4> ges;
  std::array<bool, 4> barrels;

  // Loop over triplets
  for(const auto& triplet: triplets) {
    GlobalPoint gp0 = triplet.inner()->globalPosition();
    GlobalPoint gp1 = triplet.middle()->globalPosition();
    GlobalPoint gp2 = triplet.outer()->globalPosition();

    PixelRecoLineRZ line(gp0, gp2);
    ThirdHitPredictionFromCircle predictionRPhi(gp0, gp2, extraHitRPhitolerance);

    const double curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gp1.x(), gp1.y()));

    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);
    const float thisExtraPhiTolerance = extraPhiToleranceEval.value(abscurv);

    constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f); // ...and continue as before

    auto isBarrel = [](const unsigned id) -> bool {
      return id == PixelSubdetector::PixelBarrel;
    };

    gps[0] = triplet.inner()->globalPosition();
    ges[0] = triplet.inner()->globalPositionError();
    barrels[0] = isBarrel(triplet.inner()->geographicalId().subdetId());

    gps[1] = triplet.middle()->globalPosition();
    ges[1] = triplet.middle()->globalPositionError();
    barrels[1] = isBarrel(triplet.middle()->geographicalId().subdetId());

    gps[2] = triplet.outer()->globalPosition();
    ges[2] = triplet.outer()->globalPositionError();
    barrels[2] = isBarrel(triplet.outer()->geographicalId().subdetId());

    for(size_t il=0; il!=size; ++il) {
      if(hitTree[il].empty()) continue; // Don't bother if no hits

      auto const& hits = *fourthHitMap[il];
      const DetLayer *layer = fourthLayers[il].detLayer();
      bool barrelLayer = layer->isBarrel();

      auto& predictionRZ = preds[il];
      predictionRZ.initPropagator(&line);
      Range rzRange = predictionRZ(); // z in barrel, r in endcap

      // construct search box and search
      Range phiRange;
      if(barrelLayer) {
        auto radius = static_cast<const BarrelDetLayer *>(layer)->specificSurface().radius();
        double phi = predictionRPhi.phi(curvature, radius);
        phiRange = Range(phi-thisExtraPhiTolerance, phi+thisExtraPhiTolerance);
      }
      else {
        double phi1 = predictionRPhi.phi(curvature, rzRange.min());
        double phi2 = predictionRPhi.phi(curvature, rzRange.max());
        phiRange = Range(std::min(phi1, phi2)-thisExtraPhiTolerance, std::max(phi1, phi2)+thisExtraPhiTolerance);
      }

      KDTreeBox phiZ(phiRange.min(), phiRange.max(),
                     rzRange.min()-nSigmaRZ*rzError[il],
                     rzRange.max()+nSigmaRZ*rzError[il]);

      foundNodes.clear();
      hitTree[il].search(phiZ, foundNodes);

      if(foundNodes.empty()) {
        continue;
      }

      SeedingHitSet::ConstRecHitPointer selectedHit = nullptr;
      float selectedChi2 = std::numeric_limits<float>::max();
      for(auto hitIndex: foundNodes) {
        const auto& hit = hits.theHits[hitIndex].hit();

        // Reject comparitor. For now, because of technical
        // limitations, pass three hits to the comparitor
        // TODO: switch to using hits from 2-3-4 instead of 1-3-4?
        // Eventually we should fix LowPtClusterShapeSeedComparitor to
        // accept quadruplets.
        if(theComparitor) {
          SeedingHitSet tmpTriplet(triplet.inner(), triplet.outer(), hit);
          if(!theComparitor->compatible(tmpTriplet)) {
            continue;
          }
        }

        gps[3] = hit->globalPosition();
        ges[3] = hit->globalPositionError();
        barrels[3] = isBarrel(hit->geographicalId().subdetId());

        float chi2 = std::numeric_limits<float>::quiet_NaN();
        // TODO: Do we have any use case to not use bending correction?
        if(useBendingCorrection) {
          // Following PixelFitterByConformalMappingAndLine
          const float simpleCot = ( gps.back().z()-gps.front().z() )/ (gps.back().perp() - gps.front().perp() );
          const float pt = 1/PixelRecoUtilities::inversePt(abscurv, es);
          for (int i=0; i< 4; ++i) {
            const GlobalPoint & point = gps[i];
            const GlobalError & error = ges[i];
            bc_r[i] = sqrt( sqr(point.x()-region.origin().x()) + sqr(point.y()-region.origin().y()) );
            bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt,es)(bc_r[i]);
            bc_z[i] = point.z()-region.origin().z();
            bc_errZ2[i] =  (barrels[i]) ? error.czz() : error.rerr(point)*sqr(simpleCot);
          }
          RZLine rzLine(bc_r,bc_z,bc_errZ2, RZLine::ErrZ2_tag());
          chi2 = rzLine.chi2();
        }
        else {
          RZLine rzLine(gps, ges, barrels);
          chi2 = rzLine.chi2();
        }
        if(edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
          continue;
        }
        // TODO: Do we have any use case to not use circle fit? Maybe
        // HLT where low-pT inefficiency is not a problem?
        if(fitFastCircle) {
          FastCircleFit c(gps, ges);
          chi2 += c.chi2();
          if(edm::isNotFinite(chi2))
            continue;
          if(fitFastCircleChi2Cut && chi2 > thisMaxChi2)
            continue;
        }


        if(chi2 < selectedChi2) {
          selectedChi2 = chi2;
          selectedHit = hit;
        }
      }
      if(selectedHit)
        result.emplace_back(triplet.inner(), triplet.middle(), triplet.outer(), selectedHit);
    }
  }
}
