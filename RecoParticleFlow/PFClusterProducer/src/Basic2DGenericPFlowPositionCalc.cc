#include "Basic2DGenericPFlowPositionCalc.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "vdt/vdtMath.h"

void Basic2DGenericPFlowPositionCalc::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void Basic2DGenericPFlowPositionCalc::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

void Basic2DGenericPFlowPositionCalc::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  }  				
  double cl_energy = 0;  
  double cl_time = 0;  
  double cl_timeweight=0.0;
  double max_e = 0.0;  
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  reco::PFRecHitRef refseed;  
  // find the seed and max layer and also calculate time
  //Michalis : Even if we dont use timing in clustering here we should fill
  //the time information for the cluster. This should use the timing resolution(1/E)
  //so the weight should be fraction*E^2
  //calculate a simplistic depth now. The log weighted will be done
  //in different stage  
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    if( refhit->detId() == cluster.seed() ) refseed = refhit;
    const double rh_fraction = rhf.fraction();
    const double rh_rawenergy = refhit->energy();
    const double rh_energy = rh_rawenergy * rh_fraction;   
    if( edm::isNotFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit->detId() << " has a NaN energy... " 
	<< "The input of the particle flow clustering seems to be corrupted.";
    }
    cl_energy += rh_energy;
    // If time resolution is given, calculated weighted average
    if (_timeResolutionCalcBarrel && _timeResolutionCalcEndcap) {
      double res2 = 10000.;
      int cell_layer = (int)refhit->layer();
      if (cell_layer == PFLayer::HCAL_BARREL1 ||
          cell_layer == PFLayer::HCAL_BARREL2 ||
          cell_layer == PFLayer::ECAL_BARREL)
        res2 = _timeResolutionCalcBarrel->timeResolution2(rh_rawenergy);
      else
        res2 = _timeResolutionCalcEndcap->timeResolution2(rh_rawenergy);
      cl_time += rh_fraction*refhit->time()/res2;
      cl_timeweight += rh_fraction/res2;
    }
    else { // assume resolution = 1/E**2
      const double rh_rawenergy2 = rh_rawenergy*rh_rawenergy;
      cl_timeweight+=rh_rawenergy2*rh_fraction;
      cl_time += rh_rawenergy2*rh_fraction*refhit->time();
    }

    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
    }    
  }
  cluster.setEnergy(cl_energy);
  cluster.setTime(cl_time/cl_timeweight);
  cluster.setLayer(max_e_layer);
  // calculate the position

  double depth = 0.0;  
  double position_norm = 0.0;
  double x(0.0),y(0.0),z(0.0);
  const reco::PFRecHitRefVector* seedNeighbours = NULL;
  switch( _posCalcNCrystals ) {
  case 5:
    seedNeighbours = &refseed->neighbours4();
    break;
  case 9:
    seedNeighbours = &refseed->neighbours8();
    break;
  default:
    break;
  }

  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    
    if( refhit != refseed && _posCalcNCrystals != -1 ) {
      auto pos = std::find(seedNeighbours->begin(),seedNeighbours->end(),
			   refhit);
      if( pos == seedNeighbours->end() ) continue;
    }
    
    const double rh_energy = refhit->energy() * ((float)rhf.fraction());
    const double norm = ( rhf.fraction() < _minFractionInCalc ? 
			  0.0 : 
			  std::max(0.0,vdt::fast_log(rh_energy/_logWeightDenom)) );
    const math::XYZPoint& rhpos_xyz = refhit->position();
    x += rhpos_xyz.X() * norm;
    y += rhpos_xyz.Y() * norm;
    z += rhpos_xyz.Z() * norm;
    depth += refhit->depth()*norm;
    
    position_norm += norm;
  }
  if( position_norm < _minAllowedNorm ) {
    edm::LogError("WeirdClusterNormalization") 
      << "PFCluster too far from seeding cell: set position to (0,0,0).";
    cluster.setPosition(math::XYZPoint(0,0,0));
    cluster.calculatePositionREP();
  } else {
    const double norm_inverse = 1.0/position_norm;
    x *= norm_inverse;
    y *= norm_inverse;
    z *= norm_inverse;
    depth *= norm_inverse;
    cluster.setPosition(math::XYZPoint(x,y,z));
    cluster.setDepth(depth);
    cluster.calculatePositionREP();
  }
}
