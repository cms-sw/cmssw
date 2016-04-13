#include "ECAL2DPositionCalcWithDepthCorr.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "vdt/vdtMath.h"

// faithful reimplementation of RecoEcal/EgammaCoreTools PositionCalc
// sorry Stefano

void ECAL2DPositionCalcWithDepthCorr::
update(const edm::EventSetup& es) {
  
    const CaloGeometryRecord& caloGeom = es.get<CaloGeometryRecord>();
    edm::ESHandle<CaloGeometry> geohandle;
    caloGeom.get(geohandle);
    _ebGeom = geohandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
    _eeGeom = geohandle->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
    _esGeom = geohandle->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
    // ripped from RecoEcal/EgammaCoreTools 
    for( uint32_t ic = 0; 
	 ic < _esGeom->getValidDetIds().size() && 
	   ( !_esPlus || !_esMinus ); ++ic ) {
	const double z = _esGeom->getGeometry( _esGeom->getValidDetIds()[ic] )->getPosition().z();
	_esPlus = _esPlus || ( 0 < z ) ;
	_esMinus = _esMinus || ( 0 > z ) ;
    }  
  
}

void ECAL2DPositionCalcWithDepthCorr::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void ECAL2DPositionCalcWithDepthCorr::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

void ECAL2DPositionCalcWithDepthCorr::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  constexpr double preshowerStartEta =  1.653;
  constexpr double preshowerEndEta = 2.6;
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  }  				
  double cl_energy = 0;  
  double cl_energy_float = 0;
  double cl_time = 0;  
  double cl_timeweight=0.0;
  double max_e = 0.0;  
  double clusterT0 = 0.0;
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  reco::PFRecHitRef refmax;  
  // find the seed and max layer
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    const double rh_fraction = rhf.fraction();
    const double rh_rawenergy = refhit->energy();
    const double rh_energy = rh_rawenergy * rh_fraction;    
    const double rh_energyf = ((float)rh_rawenergy) * ((float) rh_fraction);
    if( !edm::isFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit->detId() << " has non-finite energy... " 
	<< "The input of the particle flow clustering seems to be corrupted.";
    }
    cl_energy += rh_energy;
    cl_energy_float += rh_energyf;
    // If time resolution is given, calculate weighted average
    if (_timeResolutionCalc) {
      const double res2 = _timeResolutionCalc->timeResolution2(rh_rawenergy);
      cl_time += rh_fraction*refhit->time()/res2;
      cl_timeweight += rh_fraction/res2;
    }
    else { // assume resolution ~ 1/E**2
      const double rh_rawenergy2 = rh_rawenergy*rh_rawenergy;
      cl_timeweight+=rh_rawenergy2*rh_fraction;
      cl_time += rh_rawenergy2*rh_fraction*refhit->time();
    }
    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
      refmax = refhit;
    }    
  }
  cluster.setEnergy(cl_energy);
  cluster.setTime(cl_time/cl_timeweight);
  cluster.setLayer(max_e_layer);
  const CaloSubdetectorGeometry* ecal_geom = NULL;
  // get seed geometry information  
  switch(max_e_layer){
  case PFLayer::ECAL_BARREL:
    ecal_geom = _ebGeom;
    clusterT0 = _param_T0_EB;
    break;
  case PFLayer::ECAL_ENDCAP:
    ecal_geom = _eeGeom;
    clusterT0 = _param_T0_EE;        
    break;
  default:
    throw cms::Exception("InvalidLayer")
      << "ECAL Position Calc only accepts ECAL_BARREL or ECAL_ENDCAP";
  }

  const CaloCellGeometry* center_cell = 
    ecal_geom->getGeometry(refmax->detId());
  const double ctreta = center_cell->getPosition().eta();
  const double actreta = std::abs(ctreta);
  // need to change T0 if in ES
  if( actreta > preshowerStartEta && actreta < preshowerEndEta ) { 
    if(ctreta > 0 && _esPlus ) clusterT0 = _param_T0_ES;
    if(ctreta < 0 && _esMinus) clusterT0 = _param_T0_ES;
  }  
  // floats to reproduce exactly the EGM code
  const float maxDepth = _param_X0*(clusterT0 + vdt::fast_log(cl_energy_float));
  const float maxToFront = center_cell->getPosition().mag();  
  // calculate the position
  const double logETot_inv = -vdt::fast_log(cl_energy_float);
  double position_norm = 0.0;
  double x(0.0),y(0.0),z(0.0);  
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    double weight = 0.0;
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    const double rh_energy = ((float)refhit->energy()) * ((float)rhf.fraction());
    if( rh_energy > 0.0 ) weight = std::max(0.0,( _param_W0 + 
						  vdt::fast_log(rh_energy) + 
						  logETot_inv ));
    const CaloCellGeometry* cell = ecal_geom->getGeometry(refhit->detId());
    const float depth = maxDepth + maxToFront - cell->getPosition().mag();    
    const GlobalPoint pos =
      static_cast<const TruncatedPyramid*>(cell)->getPosition(depth);

    x += weight*pos.x() ;
    y += weight*pos.y() ;
    z += weight*pos.z() ;

    position_norm += weight ;
  }
  
  // FALL BACK to LINEAR WEIGHTS
  if (position_norm == 0.) {
    for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
      double weight = 0.0;
      const reco::PFRecHitRef& refhit = rhf.recHitRef();
      const double rh_energy = ((float)refhit->energy()) * ((float)rhf.fraction());
      if( rh_energy > 0.0 ) 
	weight = rh_energy/cluster.energy();

      const CaloCellGeometry* cell = ecal_geom->getGeometry(refhit->detId());
      const float depth = maxDepth + maxToFront - cell->getPosition().mag();    
      const GlobalPoint pos = static_cast<const TruncatedPyramid*>(cell)->getPosition(depth);
      
      x += weight*pos.x() ;
      y += weight*pos.y() ;
      z += weight*pos.z() ;

      position_norm += weight ;
    }
  }

  if( position_norm < _minAllowedNorm ) {
    edm::LogError("WeirdClusterNormalization") 
      << "PFCluster too far from seeding cell: set position to (0,0,0).";
    cluster.setPosition(math::XYZPoint(0,0,0));
  } else {
    const double norm_inverse = 1.0/position_norm;
    x *= norm_inverse;
    y *= norm_inverse;
    z *= norm_inverse;

    cluster.setPosition(math::XYZPoint(x,y,z));
    cluster.calculatePositionREP();
  }
}
