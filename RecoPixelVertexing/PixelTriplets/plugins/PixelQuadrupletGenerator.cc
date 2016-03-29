#include "PixelQuadrupletGenerator.h"
#include "ThirdHitRZPrediction.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "RecoPixelVertexing/PixelTrackFitting/src/RZLine.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

typedef PixelRecoRange<float> Range;

PixelQuadrupletGenerator::PixelQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")), //extra window in ThirdHitRZPrediction range
  extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
  maxChi2(cfg.getParameter<double>("maxChi2")),
  keepTriplets(cfg.getParameter<bool>("keepTriplets"))
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


  // Loop over triplets
  for(const auto& triplet: triplets) {
    GlobalPoint gp0 = triplet.inner()->globalPosition();
    GlobalPoint gp1 = triplet.middle()->globalPosition();
    GlobalPoint gp2 = triplet.outer()->globalPosition();

    PixelRecoLineRZ line(gp0, gp2);
    ThirdHitPredictionFromCircle predictionRPhi(gp0, gp2, extraHitRPhitolerance);

    const double curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gp1.x(), gp1.y()));

    constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f); // ...and continue as before

    SeedingHitSet::ConstRecHitPointer selectedHit = nullptr;
    float selectedChi2 = std::numeric_limits<float>::max();
    unsigned nLayersWithManyHits = 0;

    auto isBarrel = [](const unsigned id) -> bool {
      return id == PixelSubdetector::PixelBarrel;
    };

    std::vector<GlobalPoint> gps(4);
    std::vector<GlobalError> ges(4);
    std::vector<bool> barrels(4);
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
        //double tol = extraHitRPhitolerance/radius;
        const double tol = 0.15;
        phiRange = Range(phi-tol, phi+tol);
      }
      else {
        double phi1 = predictionRPhi.phi(curvature, rzRange.min());
        double phi2 = predictionRPhi.phi(curvature, rzRange.max());
        const double tol = 0.15;
        phiRange = Range(std::min(phi1, phi2)-tol, std::max(phi1, phi2)+tol);
      }

      KDTreeBox phiZ(phiRange.min(), phiRange.max(),
                     rzRange.min()-nSigmaRZ*rzError[il],
                     rzRange.max()+nSigmaRZ*rzError[il]);

      foundNodes.clear();
      hitTree[il].search(phiZ, foundNodes);

      if(foundNodes.empty()) {
        continue;
      }

      std::vector<std::tuple<unsigned int, float> > passedQualityCuts; // hit index, chi2
      constexpr unsigned kHitIndex = 0;
      constexpr unsigned kChi2 = 1;
      for(auto hitIndex: foundNodes) {
        const auto& hit = hits.theHits[hitIndex].hit();

        // Reject comparitor. For now, because of technical
        // limitations, pass three hits to the comparitor
        if(theComparitor) {
          SeedingHitSet tmpTriplet(triplet.inner(), triplet.outer(), hit);
          if(!theComparitor->compatible(tmpTriplet, region)) {
            continue;
          }
        }

        gps[3] = hit->globalPosition();
        ges[3] = hit->globalPositionError();
        barrels[3] = isBarrel(hit->geographicalId().subdetId());

        RZLine rzLine(gps, ges, barrels);
        float  cottheta, intercept, covss, covii, covsi;
        rzLine.fit(cottheta, intercept, covss, covii, covsi);
        float chi2 = rzLine.chi2(cottheta, intercept);
        if(edm::isNotFinite(chi2) || chi2 > maxChi2) {
          continue;
        }

        passedQualityCuts.push_back(std::make_tuple(hitIndex, chi2));
      }

      if(passedQualityCuts.empty()) {
        continue;
      }

      if(passedQualityCuts.size() == 1) {
        unsigned index = std::get<kHitIndex>(passedQualityCuts[0]);
        const auto& hit = hits.theHits[index].hit();
        float chi2 = std::get<kChi2>(passedQualityCuts[0]);

        if(chi2 < selectedChi2) {
          selectedHit = hit;
          selectedChi2 = chi2;
        }
      }
      else {
        ++nLayersWithManyHits;
      }
    }

    if(nLayersWithManyHits == 0) {
      if(selectedHit) {
        SeedingHitSet quadruplet(triplet.inner(), triplet.middle(), triplet.outer(), selectedHit);
        result.push_back(quadruplet);
      }
    }
    else if(keepTriplets) {
      result.push_back(triplet);
    }
  }
}
