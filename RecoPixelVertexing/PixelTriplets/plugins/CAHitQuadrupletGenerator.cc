#include "CAHitQuadrupletGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "LayerQuadruplets.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"


#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"



#include "CellularAutomaton.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

#include <functional>

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

constexpr unsigned int CAHitQuadrupletGenerator::minLayers;

CAHitQuadrupletGenerator::CAHitQuadrupletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC, bool needSeedingLayerSetsHits) :
extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
caThetaCut(cfg.getParameter<double>("CAThetaCut")),
caPhiCut(cfg.getParameter<double>("CAPhiCut")),
caHardPtCut(cfg.getParameter<double>("CAHardPtCut")),
caOnlyOneLastHitPerLayerFilter(cfg.getParameter<bool>("CAOnlyOneLastHitPerLayerFilter"))
{
  if(needSeedingLayerSetsHits)
    theSeedingLayerToken = iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers"));

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

void CAHitQuadrupletGenerator::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  desc.add<bool>("CAOnlyOneLastHitPerLayerFilter",false);
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything(); // until we have moved SeedComparitor too to EDProducers
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGenerator::initEvent(const edm::Event& ev, const edm::EventSetup& es) {
  if (theComparitor) theComparitor->init(ev, es);
}

namespace {
  template <typename T_HitDoublets, typename T_GeneratorOrPairsFunction>
  void fillGraph(const SeedingLayerSetsHits& layers, CAGraph& g, T_HitDoublets& hitDoublets,
                 T_GeneratorOrPairsFunction generatorOrPairsFunction) {
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		for (unsigned int j = 0; j < 4; ++j)
		{
			auto vertexIndex = 0;
			auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
					layers[i][j].name());
			if (foundVertex == g.theLayers.end())
			{
				g.theLayers.emplace_back(layers[i][j].name(), layers[i][j].hits().size());
				vertexIndex = g.theLayers.size() - 1;
			}
			else
			{
				vertexIndex = foundVertex - g.theLayers.begin();
			}
			if (j == 0)
			{

				if (std::find(g.theRootLayers.begin(), g.theRootLayers.end(),
						vertexIndex) == g.theRootLayers.end())
				{
					g.theRootLayers.emplace_back(vertexIndex);

				}

			}
			else
			{

				auto innerVertex = std::find(g.theLayers.begin(),
						g.theLayers.end(), layers[i][j - 1].name());

				CALayerPair tmpInnerLayerPair(innerVertex - g.theLayers.begin(),
						vertexIndex);

				if (std::find(g.theLayerPairs.begin(), g.theLayerPairs.end(),
						tmpInnerLayerPair) == g.theLayerPairs.end())
				{
					const bool nonEmpty = generatorOrPairsFunction(layers[i][j-1], layers[i][j], hitDoublets);
                                        if(nonEmpty) {
                                          g.theLayerPairs.push_back(tmpInnerLayerPair);
                                          g.theLayers[vertexIndex].theInnerLayers.push_back(
							innerVertex - g.theLayers.begin());
                                          innerVertex->theOuterLayers.push_back(vertexIndex);
                                          g.theLayers[vertexIndex].theInnerLayerPairs.push_back(
							g.theLayerPairs.size() - 1);
                                          innerVertex->theOuterLayerPairs.push_back(
							g.theLayerPairs.size() - 1);
                                        }
				}

			}

		}
	}
  }
}


void CAHitQuadrupletGenerator::hitQuadruplets(const TrackingRegion& region,
		OrderedHitSeeds & result, const edm::Event& ev,
		const edm::EventSetup& es)
{
	edm::Handle<SeedingLayerSetsHits> hlayers;
	ev.getByToken(theSeedingLayerToken, hlayers);
	const SeedingLayerSetsHits& layers = *hlayers;
	if (layers.numberOfLayersInSet() != 4)
		throw cms::Exception("Configuration")
				<< "CAHitQuadrupletsGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 4, got "
				<< layers.numberOfLayersInSet();

	CAGraph g;


	std::vector<HitDoublets> hitDoublets;


	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
        fillGraph(layers, g, hitDoublets,
                  [&](const SeedingLayerSetsHits::SeedingLayer& inner,
                      const SeedingLayerSetsHits::SeedingLayer& outer,
                      std::vector<HitDoublets>& hitDoublets) {
            hitDoublets.emplace_back(thePairGenerator.doublets(region, ev, es, inner, outer));
            return true;
          });

	if (theComparitor)
		theComparitor->init(ev, es);

        std::vector<const HitDoublets *> hitDoubletsPtr;
        hitDoubletsPtr.reserve(hitDoublets.size());
        for(const auto& e: hitDoublets)
          hitDoubletsPtr.emplace_back(&e);

        hitQuadruplets(region, result, hitDoubletsPtr, g, es);
        theLayerCache.clear();
}

void CAHitQuadrupletGenerator::hitNtuplets(const IntermediateHitDoublets::RegionLayerSets& regionLayerPairs,
                                              OrderedHitSeeds& result,
                                              const edm::EventSetup& es,
                                              const SeedingLayerSetsHits& layers) {
  CAGraph g;

  std::vector<const HitDoublets *> hitDoublets;

  auto layerPairEqual = [](const IntermediateHitDoublets::LayerPairHitDoublets& pair,
                           SeedingLayerSetsHits::LayerIndex inner,
                           SeedingLayerSetsHits::LayerIndex outer) {
    return pair.innerLayerIndex() == inner && pair.outerLayerIndex() == outer;
  };
  fillGraph(layers, g, hitDoublets,
            [&](const SeedingLayerSetsHits::SeedingLayer& inner,
                const SeedingLayerSetsHits::SeedingLayer& outer,
                std::vector<const HitDoublets *>& hitDoublets) {
      using namespace std::placeholders;
      auto found = std::find_if(regionLayerPairs.begin(), regionLayerPairs.end(),
                                std::bind(layerPairEqual, _1, inner.index(), outer.index()));
      if(found != regionLayerPairs.end()) {
        hitDoublets.emplace_back(&(found->doublets()));
        return true;
      }
      return false;
    });

  hitQuadruplets(regionLayerPairs.region(), result, hitDoublets, g, es);
}

void CAHitQuadrupletGenerator::hitQuadruplets(const TrackingRegion& region,
                                              OrderedHitSeeds & result,
                                              std::vector<const HitDoublets *>& hitDoublets, const CAGraph& g,
                                              const edm::EventSetup& es) {
	//Retrieve tracker topology from geometry
	edm::ESHandle<TrackerTopology> tTopoHand;
	es.get<TrackerTopologyRcd>().get(tTopoHand);
	const TrackerTopology *tTopo=tTopoHand.product();	
	
	const int numberOfHitsInNtuplet = 4;
	std::vector<CACell::CAntuplet> foundQuadruplets;

	CellularAutomaton ca(g);

	ca.createAndConnectCells(hitDoublets, region, caThetaCut,
			caPhiCut, caHardPtCut);

	ca.evolve(numberOfHitsInNtuplet);

	ca.findNtuplets(foundQuadruplets, numberOfHitsInNtuplet);


	const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

  // re-used thoughout, need to be vectors because of RZLine interface
  std::array<float, 4> bc_r;
  std::array<float, 4> bc_z;
  std::array<float, 4> bc_errZ2;
  std::array<GlobalPoint, 4> gps;
  std::array<GlobalError, 4> ges;
  std::array<bool, 4> barrels;
  unsigned int fourthLayerId = 0;
  unsigned int previousfourthLayerId = 0;
  int subDetId = 0; 
  int previousSubDetId = 0;
  unsigned int sideId = 0; 
  unsigned int previousSideId = 0; 
  std::array<unsigned int, 2> previousCellIds ={{0,0}};
  bool isTheSameTriplet = false;
  bool isTheSameFourthLayer = false;
  bool hasAlreadyPushedACandidate = false;
  float selectedChi2 = std::numeric_limits<float>::max();


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

    if(caOnlyOneLastHitPerLayerFilter)
    {
            fourthLayerId = tTopo->layer(ahit->geographicalId());
            sideId = tTopo->side(ahit->geographicalId());
            subDetId = ahit->geographicalId().subdetId();
	    isTheSameTriplet = (quadId != 0) && (foundQuadruplets[quadId][0]->getCellId() ==  previousCellIds[0]) && (foundQuadruplets[quadId][1]->getCellId() ==  previousCellIds[1]);
            isTheSameFourthLayer = (quadId != 0) &&  (fourthLayerId == previousfourthLayerId) && (subDetId == previousSubDetId) && (sideId == previousSideId);

	    previousCellIds = {{foundQuadruplets[quadId][0]->getCellId(), foundQuadruplets[quadId][1]->getCellId()}};
	    previousfourthLayerId = fourthLayerId;


	    if(!(isTheSameTriplet && isTheSameFourthLayer ))
	    {
		selectedChi2 = std::numeric_limits<float>::max();
		hasAlreadyPushedACandidate = false;
	    }

    }


    // TODO:
    // - 'line' is not used for anything
    // - if we decide to always do the circle fit for 4 hits, we don't
    //   need ThirdHitPredictionFromCircle for the curvature; then we
    //   could remove extraHitRPhitolerance configuration parameter
    PixelRecoLineRZ line(gps[0], gps[2]);
    ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2], extraHitRPhitolerance);
    const float curvature = predictionRPhi.curvature(ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);

    if (theComparitor)
    {
      SeedingHitSet tmpTriplet(foundQuadruplets[quadId][0]->getInnerHit(), foundQuadruplets[quadId][2]->getInnerHit(), foundQuadruplets[quadId][2]->getOuterHit());


      if (!theComparitor->compatible(tmpTriplet) )
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
    } 
    else
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

    if(caOnlyOneLastHitPerLayerFilter)
    {
	    if (chi2 < selectedChi2)
	    {
		selectedChi2 = chi2;

		if(hasAlreadyPushedACandidate)
		{
			result.pop_back();

		}
		result.emplace_back(foundQuadruplets[quadId][0]->getInnerHit(), foundQuadruplets[quadId][1]->getInnerHit(),
				foundQuadruplets[quadId][2]->getInnerHit(), foundQuadruplets[quadId][2]->getOuterHit());
		hasAlreadyPushedACandidate = true;

	    }
    }
    else
    {

       result.emplace_back(foundQuadruplets[quadId][0]->getInnerHit(), foundQuadruplets[quadId][1]->getInnerHit(), foundQuadruplets[quadId][2]->getInnerHit(), foundQuadruplets[quadId][2]->getOuterHit());
    }
     
  }

}


