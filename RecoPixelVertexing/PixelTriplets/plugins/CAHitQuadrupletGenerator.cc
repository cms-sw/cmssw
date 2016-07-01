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
  std::cout << "yuhuuuuu!" << std::endl;
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

  ca.create_and_connect_cells (layersDoublets, fourLayers, region, CAThetaCut, CAPhiCut);

  ca.evolve();

  ca.find_ntuplets(foundQuadruplets, 4);

  const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);


  // re-used thoughout, need to be vectors because of RZLine interface
  std::vector<float> bc_r(4), bc_z(4), bc_errZ(4);

  declareDynArray(GlobalPoint, 4, gps);
  declareDynArray(GlobalError, 4, ges);
  declareDynArray(bool, 4, barrels);

  unsigned int numberOfFoundQuadruplets = foundQuadruplets.size();

  //  std::cout << "entering the found quadruplets loop" << std::endl;
  // Loop over quadruplets
  for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId)
  {

    //    std::cout << "checking quadruplet " << quadId << " with outer hit id " << foundQuadruplets[quadId][2]->get_outer_hit_id() << std::endl;
    auto isBarrel = [](const unsigned id) -> bool
    {
      return id == PixelSubdetector::PixelBarrel;
    };

    gps[0] = foundQuadruplets[quadId][0]->get_inner_hit()->globalPosition();
    ges[0] = foundQuadruplets[quadId][0]->get_inner_hit()->globalPositionError();
    barrels[0] = isBarrel(foundQuadruplets[quadId][0]->get_inner_hit()->geographicalId().subdetId());

    gps[1] = foundQuadruplets[quadId][1]->get_inner_hit()->globalPosition();
    ges[1] = foundQuadruplets[quadId][1]->get_inner_hit()->globalPositionError();
    barrels[1] = isBarrel(foundQuadruplets[quadId][1]->get_inner_hit()->geographicalId().subdetId());

    gps[2] = foundQuadruplets[quadId][2]->get_inner_hit()->globalPosition();
    ges[2] = foundQuadruplets[quadId][2]->get_inner_hit()->globalPositionError();
    barrels[2] = isBarrel(foundQuadruplets[quadId][2]->get_inner_hit()->geographicalId().subdetId());

    gps[3] = foundQuadruplets[quadId][2]->get_outer_hit()->globalPosition();
    ges[3] = foundQuadruplets[quadId][2]->get_outer_hit()->globalPositionError();
    barrels[3] = isBarrel(foundQuadruplets[quadId][2]->get_outer_hit()->geographicalId().subdetId());

    PixelRecoLineRZ line(gps[0], gps[2]);
    ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2], extraHitRPhitolerance);
    const float curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);

    if (theComparitor)
    {
      SeedingHitSet tmpTriplet(foundQuadruplets[quadId][0]->get_inner_hit(), foundQuadruplets[quadId][2]->get_inner_hit(), foundQuadruplets[quadId][2]->get_outer_hit());


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
      const float pt = 1 / PixelRecoUtilities::inversePt(abscurv, es);
      for (int i=0; i < 4; ++i)
      {
        const GlobalPoint & point = gps[i];
        const GlobalError & error = ges[i];
        bc_r[i] = sqrt( sqr(point.x() - region.origin().x()) + sqr(point.y() - region.origin().y()) );
        bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(bc_r[i]);
        bc_z[i] = point.z() - region.origin().z();
        bc_errZ[i] =  (barrels[i]) ? sqrt(error.czz()) : sqrt( error.rerr(point) ) * simpleCot;
      }
      RZLine rzLine(bc_r, bc_z, bc_errZ);
      float      cottheta, intercept, covss, covii, covsi;
      rzLine.fit(cottheta, intercept, covss, covii, covsi);
      chi2 = rzLine.chi2(cottheta, intercept);
    } else
    {
      RZLine rzLine(gps, ges, barrels);
      float  cottheta, intercept, covss, covii, covsi;
      rzLine.fit(cottheta, intercept, covss, covii, covsi);
      chi2 = rzLine.chi2(cottheta, intercept);
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

    //    std::cout << "it has chi2: " << chi2 << std::endl;

    result.emplace_back(foundQuadruplets[quadId][0]->get_inner_hit(), foundQuadruplets[quadId][1]->get_inner_hit(), foundQuadruplets[quadId][2]->get_inner_hit(), foundQuadruplets[quadId][2]->get_outer_hit());

  }

}
