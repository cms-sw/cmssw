#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "vdt/vdtMath.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"

class HGCLayerSpecificLogWeightedPositionCalc : public PFCPositionCalculatorBase {
 public:
  HGCLayerSpecificLogWeightedPositionCalc(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf),    
    _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
    _logWeightDenom(conf.getParameter<double>("logWeightDenominator")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")),
    _w0PerLayer(conf.getParameter<std::vector<double> >("w0PerLayer"))

  {  
    _layer2mip[(unsigned)PFLayer::HGC_ECAL]  = 55.1*1e-6;
    _layer2mip[(unsigned)PFLayer::HGC_HCALF] = 85.0*1e-6;
    _layer2mip[(unsigned)PFLayer::HGC_HCALB] = 1498.4*1e-6; 
    _timeResolutionCalcBarrel.reset(NULL);
    if( conf.exists("timeResolutionCalcBarrel") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcBarrel");
        _timeResolutionCalcBarrel.reset(new ECALRecHitResolutionProvider(timeResConf));
    }
    _timeResolutionCalcEndcap.reset(NULL);
    if( conf.exists("timeResolutionCalcEndcap") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcEndcap");
        _timeResolutionCalcEndcap.reset(new ECALRecHitResolutionProvider(timeResConf));
    }
  }
  HGCLayerSpecificLogWeightedPositionCalc(const HGCLayerSpecificLogWeightedPositionCalc&) = delete;
  HGCLayerSpecificLogWeightedPositionCalc& operator=(const HGCLayerSpecificLogWeightedPositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:
  const int _posCalcNCrystals;
  const double _logWeightDenom;
  const double _minAllowedNorm;
  const std::vector<double> _w0PerLayer;
  
  std::unordered_map<unsigned,double> _layer2mip;

  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalcBarrel;
  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalcEndcap;

  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  HGCLayerSpecificLogWeightedPositionCalc,
		  "HGCLayerSpecificLogWeightedPositionCalc");

void HGCLayerSpecificLogWeightedPositionCalc::
calculateAndSetPosition(reco::PFCluster& cluster) {
  calculateAndSetPositionActual(cluster);
}

void HGCLayerSpecificLogWeightedPositionCalc::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    calculateAndSetPositionActual(cluster);
  }
}

void HGCLayerSpecificLogWeightedPositionCalc::
calculateAndSetPositionActual(reco::PFCluster& cluster) const {  
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  }  				
  double cl_energy = 0;  
  double cl_time = 0;  
  double cl_timeweight=0.0;
  double nb_energy = 0.0;
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
  }

  if( refseed.isNull() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << "Cluster has no seed!" << std::endl;
  }
  
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
    const double rh_fraction = rhf.fraction();
    const double rh_rawenergy = refhit->energy()/_layer2mip.find((unsigned)refhit->layer())->second;
    const double rh_energy = rh_rawenergy * rh_fraction;   
    if( edm::isNotFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit->detId() << " has a NaN energy... " 
	<< "The input of the particle flow clustering seems to be corrupted.";
    }
    
    if( refhit != refseed && _posCalcNCrystals != -1 ) {
      auto pos = std::find(seedNeighbours->begin(),seedNeighbours->end(),
                           refhit);
      if( pos != seedNeighbours->end() ) {
	nb_energy += rh_energy;
      }
    } else if ( refhit == refseed ) {
      nb_energy += rh_energy;
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

  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();

    // since this is 2D position calc only use neighbours in the same layer
    //if( refhit->depth() != refseed->depth() ) continue;
    
    if( refhit != refseed && _posCalcNCrystals != -1 ) {
      auto pos = std::find(seedNeighbours->begin(),seedNeighbours->end(),
			   refhit);
      if( pos == seedNeighbours->end() ) continue;
    }
    
    const double rh_energy = refhit->energy() * ((float)rhf.fraction())/_layer2mip.find((unsigned)refhit->layer())->second; // in mips
    double norm = 1.0;
    if( refhit->layer() == PFLayer::HGC_ECAL ) {
      HGCEEDetId temp( refhit->detId() );
      norm = ( rhf.fraction() < _minFractionInCalc ? 
	       0.0 : 
	       std::max(0.0,_w0PerLayer[temp.layer()-1]+vdt::fast_log(rh_energy/nb_energy) ) );
    } else {
      norm = ( rhf.fraction() < _minFractionInCalc ? 
	       0.0 : 
	       std::max(0.0,vdt::fast_log(rh_energy)) );
    }
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
