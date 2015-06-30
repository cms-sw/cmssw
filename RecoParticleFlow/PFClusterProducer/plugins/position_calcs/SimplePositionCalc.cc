#include "SimplePositionCalc.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "vdt/vdtMath.h"

void SimplePositionCalc::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void SimplePositionCalc::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

void SimplePositionCalc::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  /*
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  } 
  */
  double cl_energy = 0;  
  double max_e = 0.0;  
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  reco::PFRecHitRef refseed;  
  // find the seed and max layer and also calculate time
  //Michalis : Even if we dont use timing in clustering here we should fill
  //the time information for the cluster. This should use the timing resolution(1/E)
  //so the weight should be fraction*E^2
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

    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
    }    
  }
  cluster.setEnergy(cl_energy);
  //cluster.setTime(cl_time/cl_timeweight);
  cluster.setLayer(max_e_layer);
  // calculate the position
  double position_norm = 0.0;
  double x(0.0),y(0.0),z(0.0);  

  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();    

    const double norm = 1.0;
    const math::XYZPoint& rhpos_xyz = refhit->position();
    x += rhpos_xyz.X() * norm;
    y += rhpos_xyz.Y() * norm;
    z += rhpos_xyz.Z() * norm;
    position_norm += norm;
  }
  if( position_norm < 1 ) {
    edm::LogError("WeirdClusterNormalization") 
      << "PFCluster too far from seeding cell: set position to (0,0,0).";
    cluster.setPosition(math::XYZPoint(0,0,0));
    cluster.calculatePositionREP();
  } else {
    const double norm_inverse = 1.0/position_norm;
    x *= norm_inverse;
    y *= norm_inverse;
    z *= norm_inverse;
    cluster.setPosition(math::XYZPoint(x,y,z));
    cluster.calculatePositionREP();
  }
}
