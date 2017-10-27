#include "Basic2DGenericPFlowPositionCalc.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include "CommonTools/Utils/interface/DynArray.h"
#include<iterator>
#include <boost/function_output_iterator.hpp>

#include "vdt/vdtMath.h"


namespace {
  inline
  bool isBarrel(int cell_layer){
    return (cell_layer == PFLayer::HCAL_BARREL1 ||
	    cell_layer == PFLayer::HCAL_BARREL2 ||
	    cell_layer == PFLayer::ECAL_BARREL);
  }  
}

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
  // find the seed and max layer and also calculate time
  //Michalis : Even if we dont use timing in clustering here we should fill
  //the time information for the cluster. This should use the timing resolution(1/E)
  //so the weight should be fraction*E^2
  //calculate a simplistic depth now. The log weighted will be done
  //in different stage  


  auto const recHitCollection = &(*cluster.recHitFractions()[0].recHitRef()) - cluster.recHitFractions()[0].recHitRef().key();
  auto nhits = cluster.recHitFractions().size();
  struct LHit{ reco::PFRecHit const * hit; float energy; float fraction;};
  declareDynArray(LHit,nhits,hits);
  for(auto i=0U; i<nhits; ++i) {
    auto const & hf = cluster.recHitFractions()[i];
    auto k = hf.recHitRef().key();
    auto p = recHitCollection+k;
    hits[i]= {p,(*p).energy(), float(hf.fraction())}; 
  }


  bool resGiven = bool(_timeResolutionCalcBarrel) & bool(_timeResolutionCalcEndcap);
  LHit mySeed={nullptr};
  for( auto const & rhf : hits ) {
    const reco::PFRecHit & refhit = *rhf.hit;
    if( refhit.detId() == cluster.seed() )  mySeed = rhf;
    const auto rh_fraction = rhf.fraction;
    const auto rh_rawenergy = rhf.energy;
    const auto rh_energy = rh_rawenergy * rh_fraction;   
#ifdef PF_DEBUG
    if unlikely( edm::isNotFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit.detId() << " has a NaN energy... " 
	<< "The input of the particle flow clustering seems to be corrupted.";
    }
#endif
    cl_energy += rh_energy;
    // If time resolution is given, calculated weighted average
    if ( resGiven ) {
      double res2 = 1.e-4;
      int cell_layer = (int)refhit.layer();
      res2 =  isBarrel(cell_layer) ? 1./_timeResolutionCalcBarrel->timeResolution2(rh_rawenergy) :
                                     1./_timeResolutionCalcEndcap->timeResolution2(rh_rawenergy);
      cl_time += rh_fraction*refhit.time()*res2;
      cl_timeweight += rh_fraction*res2;
    }
    else { // assume resolution = 1/E**2
      const double rh_rawenergy2 = rh_rawenergy*rh_rawenergy;
      cl_timeweight+=rh_rawenergy2*rh_fraction;
      cl_time += rh_rawenergy2*rh_fraction*refhit.time();
    }

    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = refhit.layer();
    }    
  }
  
  cluster.setEnergy(cl_energy);
  cluster.setTime(cl_time/cl_timeweight);
  if (resGiven) {
    cluster.setTimeError(std::sqrt(1.0f/float(cl_timeweight)));
  }
  cluster.setLayer(max_e_layer);

  // calculate the position
  double depth = 0.0;  
  double position_norm = 0.0;
  double x(0.0),y(0.0),z(0.0);
  if( nullptr != mySeed.hit ) {
    auto seedNeighbours = mySeed.hit->neighbours();
    switch( _posCalcNCrystals ) {
    case 5:
      seedNeighbours = mySeed.hit->neighbours4();
      break;
    case 9:
      seedNeighbours = mySeed.hit->neighbours8();
      break;
    default:
      break;
    }
    
    auto compute = [&](LHit const& rhf) {
      const reco::PFRecHit & refhit = *rhf.hit;  
      const auto rh_energy = rhf.energy * rhf.fraction;
      const auto norm = ( rhf.fraction < _minFractionInCalc ? 
			  0.0f : 
			  std::max(0.0f,vdt::fast_logf(rh_energy*_logWeightDenom)) );
      const auto rhpos_xyz = refhit.position()*norm;
      x += rhpos_xyz.x();
      y += rhpos_xyz.y();
      z += rhpos_xyz.z();
      depth += refhit.depth()*norm;
      position_norm += norm;
    };
    
    if(_posCalcNCrystals != -1) // sorted to make neighbour search faster (maybe)
      std::sort(hits.begin(),hits.end(),[](LHit const& a, LHit const& b) { return a.hit<b.hit;});
    
    
    if(_posCalcNCrystals == -1)
      for( auto const & rhf : hits ) compute(rhf);
    else {  // only seed and its neighbours
      compute(mySeed);
      // search seedNeighbours to find energy fraction in cluster (sic)
      unInitDynArray(reco::PFRecHit const *,seedNeighbours.size(),nei);	  
      for(auto k : seedNeighbours){ 
	nei.push_back(recHitCollection+k);
      }
      std::sort(nei.begin(),nei.end());
      struct LHitLess {
	auto operator()(LHit const &a, reco::PFRecHit const * b) const {return a.hit<b;}
	auto operator()(reco::PFRecHit const * b, LHit const &a) const {return b<a.hit;}
      };
      std::set_intersection(hits.begin(),hits.end(),nei.begin(),nei.end(), 
			    boost::make_function_output_iterator(compute),
			    LHitLess()
			    );
    }
  } else {
    throw cms::Exception("Basic2DGenerticPFlowPositionCalc")
      << "Cluster seed hit is null, something is wrong with PFlow RecHit!";
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
