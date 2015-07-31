#include "Semi3DPositionCalc.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <map>
#include <tuple>

#include "vdt/vdtMath.h"

void Semi3DPositionCalc::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void Semi3DPositionCalc::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

void Semi3DPositionCalc::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  typedef std::tuple<double,double,double,double,double> CoordinateAndWeight;
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  } 
  // want map for free depth sorting
  std::map<int,CoordinateAndWeight> positionAtDepth; 
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
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    if( refhit->detId() == cluster.seed() ) refseed = refhit;
    const double rh_fraction = rhf.fraction();
    const double rh_rawenergy = refhit->energy();
    const double rh_energy = rh_rawenergy * rh_fraction;   
    if( edm::isNotFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit->detId() << " has a NaN energy... " 
	<< " fraction = " << rh_fraction << "; total energy = " << rh_rawenergy
	<< "; The input of the particle flow clustering seems to be corrupted.";
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
  double total_norm = 0.0;
  const reco::PFRecHitRefVector* seedNeighbours = NULL;
  if( _posCalcNCrystals != -1 ) {
    seedNeighbours = &refseed->neighbours();
  }
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();

    if( refhit != refseed && _posCalcNCrystals != -1 ) {
      auto pos = std::find(seedNeighbours->begin(),seedNeighbours->end(),
			   refhit);
      if( pos == seedNeighbours->end() ) continue;
    }

    if( positionAtDepth.find(refhit->depth()) == positionAtDepth.end() ) {
      positionAtDepth[refhit->depth()] = std::make_tuple(0.0,0.0,0.0,0.0,0.0);
    }
    const double rh_energy = refhit->energy() * ((float)rhf.fraction());
    const double norm = ( rhf.fraction() < _minFractionInCalc ? 
			  0.0 : 
			  std::max(0.0,vdt::fast_log(rh_energy/_logWeightDenom)) );    
    const math::XYZPoint& rhpos_xyz = refhit->position();
    const reco::PFCluster::REPPoint rhpos_rep(rhpos_xyz);
    std::get<0>(positionAtDepth[refhit->depth()]) += rhpos_xyz.X() * norm;
    std::get<1>(positionAtDepth[refhit->depth()]) += rhpos_xyz.Y() * norm;
    std::get<2>(positionAtDepth[refhit->depth()]) += rhpos_xyz.Z() * norm;
    std::get<3>(positionAtDepth[refhit->depth()]) += norm;
    std::get<4>(positionAtDepth[refhit->depth()]) += rh_energy;
    total_norm += norm;
  }
  if( total_norm < _minAllowedNorm ) {
    edm::LogError("WeirdClusterNormalization") 
      << "PFCluster too far from seeding cell: set position to (0,0,0).";
    cluster.setPosition(math::XYZPoint(0,0,0));
    cluster.calculatePositionREP();
  } else {
    // here it is Eta/Phi/Rho/weight
    CoordinateAndWeight total_position = std::make_tuple(0.0,0.0,1e6,0.0,cluster.energy());
    for( auto depth : positionAtDepth ) {
      if( std::get<3>(depth.second) >= _minAllowedNorm ) {
	const double norm_inverse = 1.0/std::get<3>(depth.second);
	const double log_depth_E = 
	  std::max(0.0,
		   vdt::fast_log(std::get<4>(depth.second)/_logWeightDenom));
	std::get<0>(depth.second) *= norm_inverse;
	std::get<1>(depth.second) *= norm_inverse;
	std::get<2>(depth.second) *= norm_inverse;
	math::XYZPoint temppos(std::get<0>(depth.second),
			       std::get<1>(depth.second),
			       std::get<2>(depth.second));
	std::get<0>(total_position) += temppos.Eta()*log_depth_E;
	std::get<1>(total_position) += temppos.Phi()*log_depth_E;
	std::get<2>(total_position) = std::min(std::get<2>(total_position),
					       temppos.R());
	std::get<3>(total_position) += log_depth_E;
      }
    }    
    const double norm_inverse = 1.0/std::get<3>(total_position);
    // get log weighted average of eta/phi across all depths
    std::get<0>(total_position) *= norm_inverse;
    std::get<1>(total_position) *= norm_inverse;
    // get the perpindicular component of R for the average
    std::get<2>(total_position) /= std::cosh(std::get<0>(total_position));
    reco::PFCluster::REPPoint pos(std::get<2>(total_position),
				  std::get<0>(total_position),
				  std::get<1>(total_position));
    cluster.setPosition(math::XYZPoint(pos.X(),pos.Y(),pos.Z()));
    cluster.calculatePositionREP();
  }
}
