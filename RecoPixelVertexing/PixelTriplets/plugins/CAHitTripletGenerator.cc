#include <unordered_map>

#include "CAHitTripletGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CellularAutomaton.h"

#include "CommonTools/Utils/interface/DynArray.h"

#include "FWCore/Utilities/interface/isFinite.h"

namespace
{

template<typename T>
T sqr(T x)
{
	return x * x;
}
}

using namespace std;
using namespace ctfseeding;

CAHitTripletGenerator::CAHitTripletGenerator(const edm::ParameterSet& cfg,
		edm::ConsumesCollector& iC) :
		theSeedingLayerToken(iC.consumes < SeedingLayerSetsHits	> (cfg.getParameter < edm::InputTag > ("SeedingLayers"))),
		extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
		maxChi2(cfg.getParameter < edm::ParameterSet > ("maxChi2")),
		useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
		caThetaCut(	cfg.getParameter<double>("CAThetaCut")),
		caPhiCut(cfg.getParameter<double>("CAPhiCut")),
		caHardPtCut(cfg.getParameter<double>("CAHardPtCut"))
{
	if (cfg.exists("SeedComparitorPSet"))
	{
		edm::ParameterSet comparitorPSet = cfg.getParameter < edm::ParameterSet
				> ("SeedComparitorPSet");
		std::string comparitorName = comparitorPSet.getParameter < std::string
				> ("ComponentName");
		if (comparitorName != "none")
		{
			theComparitor.reset(
					SeedComparitorFactory::get()->create(comparitorName,
							comparitorPSet, iC));
		}
	}
}

CAHitTripletGenerator::~CAHitTripletGenerator()
{
}

void CAHitTripletGenerator::hitTriplets(const TrackingRegion& region,
		OrderedHitTriplets & result, const edm::Event& ev,
		const edm::EventSetup& es)
{
	edm::Handle<SeedingLayerSetsHits> hlayers;
	ev.getByToken(theSeedingLayerToken, hlayers);
	const SeedingLayerSetsHits& layers = *hlayers;
	if (layers.numberOfLayersInSet() != 3)
		throw cms::Exception("Configuration")
				<< "CAHitTripletGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 3, got "
				<< layers.numberOfLayersInSet();

	CAGraph g;

	std::vector<HitDoublets> hitDoublets;

	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		for (unsigned int j = 0; j < 3; ++j)
		{
			auto vertexIndex = 0;
			auto foundVertex = std::find(g.theLayers.begin(), g.theLayers.end(),
					layers[i][j].name());
			if (foundVertex == g.theLayers.end())
			{
				g.theLayers.emplace_back(layers[i][j].name(),
						layers[i][j].hits().size());
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
					hitDoublets.emplace_back(
							thePairGenerator.doublets(region, ev, es,
									layers[i][j - 1], layers[i][j]));

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

	if (theComparitor)
		theComparitor->init(ev, es);

	std::vector<CACell::CAntuplet> foundTriplets;
	CellularAutomaton ca(g);

	ca.findTriplets(hitDoublets, foundTriplets, region, caThetaCut, caPhiCut,
			caHardPtCut);

	unsigned int numberOfFoundTriplets = foundTriplets.size();

	const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

	// re-used thoughout, need to be vectors because of RZLine interface
	std::array<float, 3> bc_r;
	std::array<float, 3> bc_z;
	std::array<float, 3> bc_errZ2;
	std::array < GlobalPoint, 3 > gps;
	std::array < GlobalError, 3 > ges;
	std::array<bool, 3> barrels;

	for (unsigned int tripletId = 0; tripletId < numberOfFoundTriplets;
			++tripletId)
	{

		OrderedHitTriplet tmpTriplet(foundTriplets[tripletId][0]->getInnerHit(),
				foundTriplets[tripletId][0]->getOuterHit(),
				foundTriplets[tripletId][1]->getOuterHit());

		auto isBarrel = [](const unsigned id) -> bool
		{
			return id == PixelSubdetector::PixelBarrel;
		};
		for (unsigned int i = 0; i < 2; ++i)
		{
			auto const& ahit = foundTriplets[tripletId][i]->getInnerHit();
			gps[i] = ahit->globalPosition();
			ges[i] = ahit->globalPositionError();
			barrels[i] = isBarrel(ahit->geographicalId().subdetId());
		}

		auto const& ahit = foundTriplets[tripletId][1]->getOuterHit();
		gps[2] = ahit->globalPosition();
		ges[2] = ahit->globalPositionError();
		barrels[2] = isBarrel(ahit->geographicalId().subdetId());

		PixelRecoLineRZ line(gps[0], gps[2]);
		ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
				extraHitRPhitolerance);
		const float curvature = predictionRPhi.curvature(
				ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
		const float abscurv = std::abs(curvature);
		const float thisMaxChi2 = maxChi2Eval.value(abscurv);
		float chi2 = std::numeric_limits<float>::quiet_NaN();
		// TODO: Do we have any use case to not use bending correction?
		if (useBendingCorrection)
		{
			// Following PixelFitterByConformalMappingAndLine
			const float simpleCot = (gps.back().z() - gps.front().z())
					/ (gps.back().perp() - gps.front().perp());
			const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
			for (int i = 0; i < 3; ++i)
			{
				const GlobalPoint & point = gps[i];
				const GlobalError & error = ges[i];
				bc_r[i] = sqrt(
						sqr(point.x() - region.origin().x())
								+ sqr(point.y() - region.origin().y()));
				bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt,
						es)(bc_r[i]);
				bc_z[i] = point.z() - region.origin().z();
				bc_errZ2[i] =
						(barrels[i]) ?
								error.czz() :
								error.rerr(point) * sqr(simpleCot);
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

		if (theComparitor)
		{
			if (!theComparitor->compatible(tmpTriplet))
			{

				continue;
			}
		}

		result.push_back(tmpTriplet);

	}
	theLayerCache.clear();
}

