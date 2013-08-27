#ifndef SeedForPhotonConversionFromQuadruplets_H
#define SeedForPhotonConversionFromQuadruplets_H

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"
#include "RecoTracker/ConversionSeedGenerators/interface/Quad.h"
class FreeTrajectoryState;

class SeedForPhotonConversionFromQuadruplets {
public:
  static const int cotTheta_Max=99999;
  
  SeedForPhotonConversionFromQuadruplets( const edm::ParameterSet & cfg):
    thePropagatorLabel(cfg.getParameter<std::string>("propagator")),
    theBOFFMomentum(cfg.existsAs<double>("SeedMomentumForBOFF") ? cfg.getParameter<double>("SeedMomentumForBOFF") : 5.0)
      {}

  SeedForPhotonConversionFromQuadruplets( 
      const std::string & propagator = "PropagatorWithMaterial", double seedMomentumForBOFF = -5.0) 
   : thePropagatorLabel(propagator), theBOFFMomentum(seedMomentumForBOFF) { }

  //dtor
  ~SeedForPhotonConversionFromQuadruplets(){}

  const TrajectorySeed * trajectorySeed( TrajectorySeedCollection & seedCollection,
						 const SeedingHitSet & phits,
						 const SeedingHitSet & mhits,
						 const TrackingRegion & region,
						 const edm::EventSetup& es,
						 std::stringstream& ss, std::vector<Quad> & quadV,
						 edm::ParameterSet& SeedComparitorPSet,
						 edm::ParameterSet& QuadCutPSet);

  
  double simpleGetSlope(const TransientTrackingRecHit::ConstRecHitPointer &ohit, const TransientTrackingRecHit::ConstRecHitPointer &nohit, const TransientTrackingRecHit::ConstRecHitPointer &ihit, const TransientTrackingRecHit::ConstRecHitPointer &nihit, const TrackingRegion & region, double & cotTheta, double & z0);
  double verySimpleFit(int size, double* ax, double* ay, double* e2y, double& p0, double& e2p0, double& p1);
  double getSqrEffectiveErrorOnZ(const TransientTrackingRecHit::ConstRecHitPointer &hit, const TrackingRegion & region);

  //
  // Some utility methods added by sguazz
  void stupidPrint(std::string s,float* d);
  void stupidPrint(std::string s,double* d);
  void stupidPrint(const char* s,GlobalPoint* d);
  void stupidPrint(const char* s,GlobalPoint* d, int n);
  void bubbleSortVsPhi(GlobalPoint arr[], int n, GlobalPoint vtx);
  void bubbleReverseSortVsPhi(GlobalPoint arr[], int n, GlobalPoint vtx);
  //
  //




 protected:

  bool checkHit(
			const TrajectoryStateOnSurface &,
			const TransientTrackingRecHit::ConstRecHitPointer &hit,
			const edm::EventSetup& es) const { return true; }

  GlobalTrajectoryParameters initialKinematic(
						      const SeedingHitSet & hits, 
						      const GlobalPoint & vertexPos, 
						      const edm::EventSetup& es,
						      const float cotTheta) const;
  
  CurvilinearTrajectoryError initialError(
						  const GlobalVector& vertexBounds, 
						  float ptMin,  
						  float sinTheta) const;
  
  const TrajectorySeed * buildSeed(
					   TrajectorySeedCollection & seedCollection,
					   const SeedingHitSet & hits,
					   const FreeTrajectoryState & fts,
					   const edm::EventSetup& es,
					   bool apply_dzCut,
					   const TrackingRegion &region) const;

  bool buildSeedBool(
      TrajectorySeedCollection & seedCollection,
      const SeedingHitSet & hits,
      const FreeTrajectoryState & fts,
      const edm::EventSetup& es,
      bool apply_dzCut,
      const TrackingRegion & region,
      double dzcut) const;
  
  TransientTrackingRecHit::RecHitPointer refitHit(
							  const TransientTrackingRecHit::ConstRecHitPointer &hit, 
							  const TrajectoryStateOnSurface &state) const;

  bool similarQuadExist(Quad & thisQuad, std::vector<Quad>& quadV);

  double DeltaPhiManual(const math::XYZVector& v1, const math::XYZVector& v2);
  

protected:
  std::string thePropagatorLabel;
  double theBOFFMomentum;
  double  kPI_;

  std::stringstream * pss;
  PrintRecoObjects po;
};
#endif 
