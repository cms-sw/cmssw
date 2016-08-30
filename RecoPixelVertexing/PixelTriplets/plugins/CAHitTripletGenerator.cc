#include <unordered_map>

#include "CAHitTripletGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTrackFitting/src/RZLine.h"
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
		theSeedingLayerToken(
				iC.consumes < SeedingLayerSetsHits> (cfg.getParameter < edm::InputTag > ("SeedingLayers"))),
				extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), //extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
				maxChi2(cfg.getParameter < edm::ParameterSet > ("maxChi2")),
				useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
				CAThetaCut(cfg.getParameter<double>("CAThetaCut")),
				CAPhiCut(cfg.getParameter<double>("CAPhiCut"))
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

	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
	std::unordered_map<std::string, HitDoublets> doubletsMap;
	std::array<const HitDoublets*, 2> layersDoublets;

	for (unsigned int j = 0; j < layers.size(); j++)
	{
		for (unsigned int i = 0; i < 2; ++i)
		{
			auto const & inner = layers[j][i];
			auto const & outer = layers[j][i + 1];
			auto layersPair = inner.name() + '+' + outer.name();
			auto it = doubletsMap.find(layersPair);
			if (it == doubletsMap.end())
			{

				std::tie(it, std::ignore) = doubletsMap.insert(
						std::make_pair(layersPair,
								thePairGenerator.doublets(region, ev, es, inner,
										outer)));
			}
			layersDoublets[i] = &it->second;

		}

		findTriplets(region, result, ev, es, layers[j], layersDoublets);
	}

	theLayerCache.clear();
}

void CAHitTripletGenerator::findTriplets(const TrackingRegion& region,
		OrderedHitTriplets& result, const edm::Event& ev,
		const edm::EventSetup& es,
		const SeedingLayerSetsHits::SeedingLayerSet& threeLayers,
		const std::array<const HitDoublets*, 2> layersDoublets)
{
//	if (theComparitor)
//		theComparitor->init(ev, es);
//	HitPairGeneratorFromLayerPair thePairGenerator(0, 1, &theLayerCache);
//
//	std::vector<CACell::CAntuplet> foundTriplets;
//
//	CellularAutomaton<3> ca;
//
//	ca.findTriplets(layersDoublets, threeLayers, foundTriplets, region,
//			CAThetaCut, CAPhiCut);
//	unsigned int numberOfFoundTriplets = foundTriplets.size();
//
//	const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);
//
//	// re-used thoughout, need to be vectors because of RZLine interface
//	std::vector<float> bc_r(3), bc_z(3), bc_errZ(3);
//
//	std::vector<GlobalPoint> gps(3);
//	std::vector<GlobalError> ges(3);
//	std::vector<bool> barrels(3);
//
//	for (unsigned int tripletId = 0; tripletId < numberOfFoundTriplets;
//			++tripletId)
//	{
//
//		OrderedHitTriplet tmpTriplet(foundTriplets[tripletId][0]->getInnerHit(),
//				foundTriplets[tripletId][0]->getOuterHit(),
//				foundTriplets[tripletId][1]->getOuterHit());
//
//
//
//
//		auto isBarrel = [](const unsigned id) -> bool
//		{
//			return id == PixelSubdetector::PixelBarrel;
//		};
//		for (unsigned int i = 0; i < 2; ++i)
//		{
//			auto const& ahit = foundTriplets[tripletId][i]->getInnerHit();
//			gps[i] = ahit->globalPosition();
//			ges[i] = ahit->globalPositionError();
//			barrels[i] = isBarrel(ahit->geographicalId().subdetId());
//		}
//
//		auto const& ahit = foundTriplets[tripletId][1]->getOuterHit();
//		gps[2] = ahit->globalPosition();
//		ges[2] = ahit->globalPositionError();
//		barrels[2] = isBarrel(ahit->geographicalId().subdetId());
//
//
//
//		PixelRecoLineRZ line(gps[0], gps[2]);
//		ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
//				extraHitRPhitolerance);
//		const float curvature = predictionRPhi.curvature(
//				ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
//		const float abscurv = std::abs(curvature);
//		const float thisMaxChi2 = maxChi2Eval.value(abscurv);
//		float chi2 = std::numeric_limits<float>::quiet_NaN();
//		// TODO: Do we have any use case to not use bending correction?
//		if (useBendingCorrection)
//		{
//			// Following PixelFitterByConformalMappingAndLine
//			const float simpleCot = (gps.back().z() - gps.front().z())
//					/ (gps.back().perp() - gps.front().perp());
//			const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
//			for (int i = 0; i < 3; ++i)
//			{
//				const GlobalPoint & point = gps[i];
//				const GlobalError & error = ges[i];
//				bc_r[i] = sqrt(
//						sqr(point.x() - region.origin().x())
//								+ sqr(point.y() - region.origin().y()));
//				bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt,
//						es)(bc_r[i]);
//				bc_z[i] = point.z() - region.origin().z();
//				bc_errZ[i] =
//						(barrels[i]) ?
//								sqrt(error.czz()) :
//								sqrt(error.rerr(point)) * simpleCot;
//			}
//			RZLine rzLine(bc_r, bc_z, bc_errZ);
//			float cottheta, intercept, covss, covii, covsi;
//			rzLine.fit(cottheta, intercept, covss, covii, covsi);
//			chi2 = rzLine.chi2(cottheta, intercept);
//		}
//		else
//		{
//			RZLine rzLine(gps, ges, barrels);
//			float cottheta, intercept, covss, covii, covsi;
//			rzLine.fit(cottheta, intercept, covss, covii, covsi);
//			chi2 = rzLine.chi2(cottheta, intercept);
//		}
//
//		if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2)
//		{
//			continue;
//
//		}
//
//		if (theComparitor)
//		{
//			if (!theComparitor->compatible(tmpTriplet, region))
//			{
//
//
//				continue;
//			}
//		}
//
//		result.push_back(tmpTriplet);
//
//
//	}
}

