/** \file SeedGeneratorForLaserBeams.cc
 *  
 *
 *  $Date: 2007/03/18 19:00:21 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/SeedGeneratorForLaserBeams.h"
#include "Alignment/LaserAlignment/interface/LaserLayerPairs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h" // really needed?
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"


	SeedGeneratorForLaserBeams::SeedGeneratorForLaserBeams(edm::ParameterSet const& iConfig) 
	: SeedGeneratorFromTrackingRegion(iConfig), conf_(iConfig), region(), 
	thePairGenerator(), magfield(), tracker(), transformer(), theUpdator(),
	thePropagatorAl(), thePropagatorOp(), TTRHBuilder(), builderName()
{
	double ptmin = conf_.getParameter<double>("ptMin");
	double originradius = conf_.getParameter<double>("originRadius");
	double halflength = conf_.getParameter<double>("originHalfLength");
	double originz = conf_.getParameter<double>("originZPosition");
	builderName = conf_.getParameter<std::string>("TTRHBuilder");

	region = GlobalTrackingRegion(ptmin, originradius, halflength, originz);

	edm::LogInfo("SeedGeneratorForLaserBeams") << " PtMin of track is " << ptmin
		<< " The Radius of the cylinder for seeds is " << originradius << " cm";
}

SeedGeneratorForLaserBeams::~SeedGeneratorForLaserBeams()
{
	if ( thePropagatorAl != 0 ) { delete thePropagatorAl; }
	if ( thePropagatorOp != 0 ) { delete thePropagatorOp; }
	if ( theUpdator != 0 ) { delete theUpdator; }
	if ( thePairGenerator != 0) { delete thePairGenerator; }
}

void SeedGeneratorForLaserBeams::init(const SiStripRecHit2DCollection &collstereo,
	const SiStripRecHit2DCollection &collrphi,
	const SiStripMatchedRecHit2DCollection &collmatched,
	const edm::EventSetup & iSetup)
{
	iSetup.get<IdealMagneticFieldRecord>().get(magfield);
	iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

	thePropagatorAl = new PropagatorWithMaterial(alongMomentum, 0.0, &(*magfield) );
	thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum, 0.0, &(*magfield) );
	theUpdator = new KFUpdator();

	// get the transient builder
	edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
	iSetup.get<TransientRecHitRecord>().get(builderName, theBuilder);
	TTRHBuilder = theBuilder.product();

	LaserLayerPairs laserLayers;
	laserLayers.init(collstereo, collrphi, collmatched, iSetup);
	thePairGenerator = new LaserHitPairGenerator(laserLayers, iSetup);
}

void SeedGeneratorForLaserBeams::run(TrajectorySeedCollection & output, const edm::EventSetup & iSetup)
{
	OrderedHitPairs HitPairs;
	thePairGenerator->hitPairs(region, HitPairs, iSetup);

	if (HitPairs.size() > 0)
	{
		stable_sort(HitPairs.begin(), HitPairs.end(), CompareHitPairsZ(iSetup) );

		for (uint i = 0; i < HitPairs.size(); i++)
		{
		  GlobalPoint inner = tracker->idToDet(HitPairs[i].inner().RecHit()->geographicalId())->surface().toGlobal(HitPairs[i].inner().RecHit()->localPosition());
		  GlobalPoint outer = tracker->idToDet(HitPairs[i].outer().RecHit()->geographicalId())->surface().toGlobal(HitPairs[i].outer().RecHit()->localPosition());

			TransientTrackingRecHit::ConstRecHitPointer outrhit = TTRHBuilder->build(HitPairs[i].outer().RecHit());

			edm::OwnVector<TrackingRecHit> hits;
			hits.push_back(HitPairs[i].inner().RecHit()->clone());
			hits.push_back(HitPairs[i].outer().RecHit()->clone());

			if ( ( (outer.z()-inner.z())>0 && outer.z() > 0 && inner.z() > 0 ) 
				|| ( (outer.z() - inner.z()) < 0 && outer.z() < 0 && inner.z() < 0 ) )
			{
				// the 0 is a possible problem!!!!
				// 	  GlobalTrajectoryParameters Gtp(outer, inner-outer, 0, &(*magfield));
				GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
				FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));

				LogDebug("LaserSeedFinder") << " FirstTSOS " << LaserSeed;

				// First propagation
				const TSOS outerState = thePropagatorAl->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer().RecHit()->geographicalId())->surface());

				if (outerState.isValid())
				{
					LogDebug("LaserSeedFinder") << " outerState " << outerState;
					const TSOS outerUpdated = theUpdator->update( outerState, *outrhit);

					if (outerUpdated.isValid())
					{
						LogDebug("LaserSeedFinder") << " outerUpdated " << outerUpdated;

						PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer().RecHit()->geographicalId().rawId());
						TrajectorySeed * trSeed = new TrajectorySeed(*pTraj, hits, alongMomentum);
					// store seed
						output.push_back(*trSeed);
					}
					else { edm::LogError("LaserSeedFinder") << " SeedForLaserBeams first update failed "; }
				}
				else { edm::LogError("LaserSeedFinder") << " SeedForLaserBeams first propagation failed "; }
			}
			else 
			{
				// the 0 is a possible problem!!!!
				// 	  GlobalTrajectoryParameters Gtp(outer, outer-inner, 0, &(*magfield));
				GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
				FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
				LogDebug("LaserSeedFinder") << " FirstTSOS " << LaserSeed;

				// First propagation
				const TSOS outerState = thePropagatorOp->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer().RecHit()->geographicalId())->surface());

				if (outerState.isValid())
				{
					LogDebug("LaserSeedFinder") << " outerState " << outerState;
					const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);

					if (outerUpdated.isValid())
					{
						LogDebug("LaserSeedFinder") << " outerUpdated " << outerUpdated;
						PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer().RecHit()->geographicalId().rawId());

						TrajectorySeed *trSeed = new TrajectorySeed(*pTraj, hits, oppositeToMomentum);
					// store seed
						output.push_back(*trSeed);
					}
					else { edm::LogError("LaserSeedFinder") << " SeedForLaserBeams first update failed "; }
				}
				else { edm::LogError("LaserSeedFinder") << " SeedForLaserBeams first propagation failed "; }
			}
		}
	}
}
