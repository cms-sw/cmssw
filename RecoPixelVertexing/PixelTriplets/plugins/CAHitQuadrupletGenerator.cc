#include "CAHitQuadrupletGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "LayerQuadruplets.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"



#include "CellularAutomaton.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

namespace
{

  template <typename T>
  T sqr(T x)
  {
    return x*x;
  }
}



using namespace std;
using namespace ctfseeding;

CAHitQuadrupletGenerator::CAHitQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) :
theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers"))),
extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
CAThetaCut(cfg.getParameter<double>("CAThetaCut")),
CAPhiCut(cfg.getParameter<double>("CAPhiCut"))
{
  if (cfg.exists("SeedComparitorPSet"))
  {
    edm::ParameterSet comparitorPSet =
            cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
    std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
    if (comparitorName != "none")
    {
      theComparitor.reset(SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));
    }
  }
}

CAHitQuadrupletGenerator::~CAHitQuadrupletGenerator()
{
}

void CAHitQuadrupletGenerator::hitQuadruplets(
                                              const TrackingRegion& region, OrderedHitSeeds & result,
                                              const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
  if (layers.numberOfLayersInSet() != 4)
    throw cms::Exception("Configuration") << "CAHitQuadrupletsGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 4, got " << layers.numberOfLayersInSet();
  for (unsigned int j=0; j < layers.size(); j++)
  {
    findQuadruplets(region, result, ev, es, layers[j]);
  }

  theLayerCache.clear();
}

void
CAHitQuadrupletGenerator::findQuadruplets (const TrackingRegion& region, OrderedHitSeeds& result,
                                           const edm::Event& ev, const edm::EventSetup& es,
                                           const SeedingLayerSetsHits::SeedingLayerSet& fourLayers)
{
  if (theComparitor) theComparitor->init (ev, es);
  HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);

  std::vector<CACell::CAntuplet> foundQuadruplets;

  std::vector<const HitDoublets*> layersDoublets(3);

  HitDoublets doublets0 =  thePairGenerator.doublets(region, ev, es, fourLayers[0], fourLayers[1] );
  HitDoublets doublets1 =  thePairGenerator.doublets(region, ev, es, fourLayers[1], fourLayers[2] );
  HitDoublets doublets2 =  thePairGenerator.doublets(region, ev, es, fourLayers[2], fourLayers[3] );

  layersDoublets[0] = &(doublets0);
  layersDoublets[1] = &(doublets1);
  layersDoublets[2] = &(doublets2);


  CellularAutomaton<4> ca;

  ca.createAndConnectCells (layersDoublets, fourLayers, region, CAThetaCut, CAPhiCut);

  ca.evolve();

  ca.findNtuplets(foundQuadruplets, 4);

  const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

  // re-used thoughout, need to be vectors because of RZLine interface
  std::array<float, 4> bc_r;
  std::array<float, 4> bc_z;
  std::array<float, 4> bc_errZ2;
  std::array<GlobalPoint, 4> gps;
  std::array<GlobalError, 4> ges;
  std::array<bool, 4> barrels;

  unsigned int numberOfFoundQuadruplets = foundQuadruplets.size();

  // Loop over quadruplets
  for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId)
  {

    auto isBarrel = [](const unsigned id) -> bool
    {
      return id == PixelSubdetector::PixelBarrel;
    };
    for(unsigned int i = 0; i< 3; ++i)
    {
        auto const& ahit = foundQuadruplets[quadId][i]->getInnerHit();
        gps[i] = ahit->globalPosition();
        ges[i] = ahit->globalPositionError();
        barrels[i] = isBarrel(ahit->geographicalId().subdetId());
    }

    auto const& ahit = foundQuadruplets[quadId][2]->getOuterHit();
    gps[3] = ahit->globalPosition();
    ges[3] = ahit->globalPositionError();
    barrels[3] = isBarrel(ahit->geographicalId().subdetId());

    PixelRecoLineRZ line(gps[0], gps[2]);
    ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2], extraHitRPhitolerance);
    const float curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);

    if (theComparitor)
    {
      SeedingHitSet tmpTriplet(foundQuadruplets[quadId][0]->getInnerHit(), foundQuadruplets[quadId][2]->getInnerHit(), foundQuadruplets[quadId][2]->getOuterHit());


      if (!theComparitor->compatible(tmpTriplet, region) )
      {
        continue;
      }
    }

    float chi2 = std::numeric_limits<float>::quiet_NaN();
    // TODO: Do we have any use case to not use bending correction?
    if (useBendingCorrection)
    {
      // Following PixelFitterByConformalMappingAndLine
      const float simpleCot = ( gps.back().z() - gps.front().z() ) / (gps.back().perp() - gps.front().perp() );
      const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
      for (int i=0; i < 4; ++i)
      {
        const GlobalPoint & point = gps[i];
        const GlobalError & error = ges[i];
        bc_r[i] = sqrt( sqr(point.x() - region.origin().x()) + sqr(point.y() - region.origin().y()) );
        bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(bc_r[i]);
        bc_z[i] = point.z() - region.origin().z();
        bc_errZ2[i] =  (barrels[i]) ? error.czz() : error.rerr(point)*sqr(simpleCot);
      }
      RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
      chi2 = rzLine.chi2();
    } else
    {
      RZLine rzLine(gps, ges, barrels);
      chi2 = rzLine.chi2();
    }
    if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2)
    {
      continue;
              
    }
    // TODO: Do we have any use case to not use circle fit? Maybe
    // HLT where low-pT inefficiency is not a problem?
    if (fitFastCircle)
    {
      FastCircleFit c(gps, ges);
      chi2 += c.chi2();
      if (edm::isNotFinite(chi2))
        continue;
      if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
        continue;
    }

    result.emplace_back(foundQuadruplets[quadId][0]->getInnerHit(), foundQuadruplets[quadId][1]->getInnerHit(), foundQuadruplets[quadId][2]->getInnerHit(), foundQuadruplets[quadId][2]->getOuterHit());
  }

}


