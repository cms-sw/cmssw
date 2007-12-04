/** \file SeedGeneratorForLaserBeams.cc
 *  
 *
 *  $Date: 2007/06/27 07:26:45 $
 *  $Revision: 1.10 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/SeedGeneratorForLaserBeams.h"
#include "FWCore/Framework/interface/EventSetup.h" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "boost/mpl/vector.hpp" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h" 
#include "Alignment/LaserAlignment/interface/LaserLayerPairs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"


	SeedGeneratorForLaserBeams::SeedGeneratorForLaserBeams(edm::ParameterSet const& iConfig) 
	  : conf_(iConfig), region(), 
	thePairGenerator(), magfield(), tracker(), transformer(), theUpdator(),
	thePropagatorMaterialAl(), thePropagatorMaterialOp(), thePropagatorAnalyticalAl(),
	thePropagatorAnalyticalOp(), TTRHBuilder(), builderName(), propagatorName()
{
	double ptmin = conf_.getParameter<double>("ptMin");
	double originradius = conf_.getParameter<double>("originRadius");
	double halflength = conf_.getParameter<double>("originHalfLength");
	double originz = conf_.getParameter<double>("originZPosition");
	builderName = conf_.getParameter<std::string>("TTRHBuilder");
  propagatorName = conf_.getParameter<std::string>("Propagator");

	region = GlobalTrackingRegion(ptmin, originradius, halflength, originz);

	edm::LogInfo("SeedGeneratorForLaserBeams") << " Using " << propagatorName 
	  << " Propagator,  PtMin of track is " << ptmin
		<< " The Radius of the cylinder for seeds is " << originradius << " cm";
}

SeedGeneratorForLaserBeams::~SeedGeneratorForLaserBeams()
{
  if (propagatorName == "WithMaterial")
  {
    if ( thePropagatorMaterialAl != 0 ) { delete thePropagatorMaterialAl; }
    if ( thePropagatorMaterialOp != 0 ) { delete thePropagatorMaterialOp; }
  }
  else if (propagatorName == "Analytical")
  {
    if ( thePropagatorAnalyticalAl != 0 ) { delete thePropagatorAnalyticalAl; }
    if ( thePropagatorAnalyticalOp != 0 ) { delete thePropagatorAnalyticalOp; }
  }	
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

  if ( propagatorName == "WithMaterial" )
  {
    thePropagatorMaterialAl = new PropagatorWithMaterial(alongMomentum, 0.0, &(*magfield) );
    thePropagatorMaterialOp = new PropagatorWithMaterial(oppositeToMomentum, 0.0, &(*magfield) );
  }	
  else if ( propagatorName == "Analytical" )
  {
    thePropagatorAnalyticalAl = new AnalyticalPropagator(&(*magfield), alongMomentum);
    thePropagatorAnalyticalOp = new AnalyticalPropagator(&(*magfield), oppositeToMomentum);
  }

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
	OrderedLaserHitPairs HitPairs;
	thePairGenerator->hitPairs(region, HitPairs, iSetup);

	if (HitPairs.size() > 0)
	{
		stable_sort(HitPairs.begin(), HitPairs.end(), CompareHitPairsZ(iSetup) );

    if (propagatorName == "WithMaterial")
    {
      propagateWithMaterial(HitPairs, output);
    }
    else if ( propagatorName == "Analytical")
    {
      propagateAnalytical(HitPairs, output);
    }
	}
}

void SeedGeneratorForLaserBeams::propagateWithMaterial(OrderedLaserHitPairs & HitPairs, TrajectorySeedCollection & output)
{
  for (uint i = 0; i < HitPairs.size(); i++)
  {
    GlobalPoint inner = tracker->idToDet(HitPairs[i].inner()->geographicalId())->surface().toGlobal(HitPairs[i].inner()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface().toGlobal(HitPairs[i].outer()->localPosition());

    TransientTrackingRecHit::ConstRecHitPointer outrhit = TTRHBuilder->build(HitPairs[i].outer());

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[i].inner()->clone());
    hits.push_back(HitPairs[i].outer()->clone());

    if ( ( (outer.z()-inner.z())>0 && outer.z() > 0 && inner.z() > 0 ) 
      || ( (outer.z() - inner.z()) < 0 && outer.z() < 0 && inner.z() < 0 ) )
    {
        // the 0 is a possible problem!!!!
        // 	  GlobalTrajectoryParameters Gtp(outer, inner-outer, 0, &(*magfield));
      GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
      FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));

      LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " FirstTSOS " << LaserSeed;

          // First propagation
      const TSOS outerState = thePropagatorMaterialAl->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface());

      if (outerState.isValid())
      {
        LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update( outerState, *outrhit);

        if (outerUpdated.isValid())
        {
          LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " outerUpdated " << outerUpdated;

          PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer()->geographicalId().rawId());
          TrajectorySeed * trSeed = new TrajectorySeed(*pTraj, hits, alongMomentum);
            // store seed
          output.push_back(*trSeed);
        }
        else { edm::LogError("SeedGeneratorForLaserBeams:propagateWithMaterial") << " SeedForLaserBeams first update failed "; }
      }
      else { edm::LogError("SeedGeneratorForLaserBeams:propagateWithMaterial") << " SeedForLaserBeams first propagation failed "; }
    }
    else 
    {
          // the 0 is a possible problem!!!!
          // 	  GlobalTrajectoryParameters Gtp(outer, outer-inner, 0, &(*magfield));
      GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
      FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
      LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " FirstTSOS " << LaserSeed;

          // First propagation
      const TSOS outerState = thePropagatorMaterialOp->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface());

      if (outerState.isValid())
      {
        LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);

        if (outerUpdated.isValid())
        {
          LogDebug("SeedGeneratorForLaserBeams:propagateWithMaterial") << " outerUpdated " << outerUpdated;
          PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer()->geographicalId().rawId());

          TrajectorySeed *trSeed = new TrajectorySeed(*pTraj, hits, oppositeToMomentum);
            // store seed
          output.push_back(*trSeed);
        }
        else { edm::LogError("SeedGeneratorForLaserBeams:propagateWithMaterial") << " SeedForLaserBeams first update failed "; }
      }
      else { edm::LogError("SeedGeneratorForLaserBeams:propagateWithMaterial") << " SeedForLaserBeams first propagation failed "; }
    }
  }
}

void SeedGeneratorForLaserBeams::propagateAnalytical(OrderedLaserHitPairs & HitPairs, TrajectorySeedCollection & output)
{
  for (uint i = 0; i < HitPairs.size(); i++)
  {
    GlobalPoint inner = tracker->idToDet(HitPairs[i].inner()->geographicalId())->surface().toGlobal(HitPairs[i].inner()->localPosition());
    GlobalPoint outer = tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface().toGlobal(HitPairs[i].outer()->localPosition());

    TransientTrackingRecHit::ConstRecHitPointer outrhit = TTRHBuilder->build(HitPairs[i].outer());

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[i].inner()->clone());
    hits.push_back(HitPairs[i].outer()->clone());

    if ( ( (outer.z()-inner.z())>0 && outer.z() > 0 && inner.z() > 0 ) 
      || ( (outer.z() - inner.z()) < 0 && outer.z() < 0 && inner.z() < 0 ) )
    {
        // the 0 is a possible problem!!!!
        // 	  GlobalTrajectoryParameters Gtp(outer, inner-outer, 0, &(*magfield));
      GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
      FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));

      LogDebug("SeedGeneratorForLaserBeams:propagateAnalytical") << " FirstTSOS " << LaserSeed;

          // First propagation
      const TSOS outerState = thePropagatorAnalyticalAl->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface());

      if (outerState.isValid())
      {
        LogDebug("SeedGeneratorForLaserBeams:propagateAnalytical") << " outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update( outerState, *outrhit);

        if (outerUpdated.isValid())
        {
          LogDebug("SeedGeneratorForLaserBeams:propagateAnalytical") << " outerUpdated " << outerUpdated;

          PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer()->geographicalId().rawId());
          TrajectorySeed * trSeed = new TrajectorySeed(*pTraj, hits, alongMomentum);
            // store seed
          output.push_back(*trSeed);
        }
        else { edm::LogError("SeedGeneratorForLaserBeams:propagateAnalytical") << " SeedForLaserBeams first update failed "; }
      }
      else { edm::LogError("SeedGeneratorForLaserBeams:propagateAnalytical") << " SeedForLaserBeams first propagation failed "; }
    }
    else 
    {
          // the 0 is a possible problem!!!!
          // 	  GlobalTrajectoryParameters Gtp(outer, outer-inner, 0, &(*magfield));
      GlobalTrajectoryParameters Gtp(outer, outer-inner, -1, &(*magfield));
      FreeTrajectoryState LaserSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix(5,1)));
      LogDebug("SeedGeneratorForLaserBeams:propagateWithAnalytical") << " FirstTSOS " << LaserSeed;

          // First propagation
      const TSOS outerState = thePropagatorAnalyticalOp->propagate(LaserSeed, tracker->idToDet(HitPairs[i].outer()->geographicalId())->surface());

      if (outerState.isValid())
      {
        LogDebug("SeedGeneratorForLaserBeams:propagateAnalytical") << " outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);

        if (outerUpdated.isValid())
        {
          LogDebug("SeedGeneratorForLaserBeams:propagateAnalytical") << " outerUpdated " << outerUpdated;
          PTrajectoryStateOnDet *pTraj = transformer.persistentState(outerUpdated, HitPairs[i].outer()->geographicalId().rawId());

          TrajectorySeed *trSeed = new TrajectorySeed(*pTraj, hits, oppositeToMomentum);
            // store seed
          output.push_back(*trSeed);
        }
        else { edm::LogError("SeedGeneratorForLaserBeams:propagateAnalytical") << " SeedForLaserBeams first update failed "; }
      }
      else { edm::LogError("SeedGeneratorForLaserBeams:propagateAnalytical") << " SeedForLaserBeams first propagation failed "; }
    }
  }
}

