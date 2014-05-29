#ifndef __CrystalCalo2DPositionCalcWithDepth_H__
#define __CrystalCalo2DPositionCalcWithDepth_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

#include <cmath>
#include <unordered_map>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "vdt/vdtMath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

/// Simple position calculation for possibly tilted crystal 
/// calorimeter geometries.
template<DetId::Detector DET,int SUBDET>
class CrystalCalo2DPositionCalcWithDepth : public PFCPositionCalculatorBase {
 public:
  CrystalCalo2DPositionCalcWithDepth(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf), 
    _param_T0(conf.getParameter<double>("T0")),
    _param_W0(conf.getParameter<double>("W0")),
    _param_X0(conf.getParameter<double>("X0")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")),
    _caloGeomRcd(NULL),
    _caloGeom(NULL) {
        _timeResolutionCalc.reset(NULL);
    if( conf.exists("timeResolutionCalc") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalc");
        _timeResolutionCalc.reset(new ECALRecHitResolutionProvider(timeResConf)); 
      }
    }
  CrystalCalo2DPositionCalcWithDepth(const CrystalCalo2DPositionCalcWithDepth&) = delete;
  CrystalCalo2DPositionCalcWithDepth& operator=(const CrystalCalo2DPositionCalcWithDepth&) = delete;

  void update(const edm::EventSetup& es);

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:  
  const double _param_T0;
  const double _param_W0;
  const double _param_X0;
  const double _minAllowedNorm;

  
  const CaloGeometryRecord* _caloGeomRcd;
  const CaloSubdetectorGeometry* _caloGeom;

  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalc;
  
  void calculateAndSetPositionActual(reco::PFCluster&) const;
};


template<DetId::Detector DET, int SUBDET>
void CrystalCalo2DPositionCalcWithDepth<DET,SUBDET>::
update(const edm::EventSetup& es) {
  const CaloGeometryRecord& temp = es.get<CaloGeometryRecord>();
  if( _caloGeomRcd == NULL || 
      ( _caloGeomRcd->cacheIdentifier() != temp.cacheIdentifier() ) ) {
    _caloGeomRcd = &temp;
    edm::ESHandle<CaloGeometry> geohandle;
    _caloGeomRcd->get(geohandle);
    _caloGeom = geohandle->getSubdetectorGeometry(DET,SUBDET);    
  }
}

template<DetId::Detector DET, int SUBDET>
void CrystalCalo2DPositionCalcWithDepth<DET,SUBDET>::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

template<DetId::Detector DET, int SUBDET>
void CrystalCalo2DPositionCalcWithDepth<DET,SUBDET>::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

template<DetId::Detector DET, int SUBDET>
void CrystalCalo2DPositionCalcWithDepth<DET,SUBDET>::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  }  				
  double cl_energy = 0;  
  double cl_energy_float = 0;
  double cl_time = 0;  
  double cl_timeweight=0.0;
  double max_e = 0.0;  
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
  const CaloSubdetectorGeometry* ecal_geom = _caloGeom;  
  const CaloCellGeometry* center_cell = 
    ecal_geom->getGeometry(refmax->detId());
  const double ctraxisangle = std::abs(static_cast<const TruncatedPyramid*>(center_cell)->getThetaAxis());
  const double showerAngleToAxis = std::abs(center_cell->getPosition().theta())-ctraxisangle;  
  // calculate tilt of calorimeter at shower max w.r.t. to (0,0,0)
  const double cosShwrAngle = std::cos(showerAngleToAxis);
  
  // floats to reproduce exactly the EGM code
  const float maxDepth = _param_X0*(_param_T0 + vdt::fast_log(cl_energy_float))*cosShwrAngle;
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

#endif
