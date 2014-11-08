#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "vdt/vdtMath.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

#include "TPrincipal.h"

class Cluster3DPCACalculator : public PFCPositionCalculatorBase {
 public:
  Cluster3DPCACalculator(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf),    
    _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
    _logWeightDenom(conf.getParameter<double>("logWeightDenominator")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization")),
    _pca(new TPrincipal(3,"D")){ }
  Cluster3DPCACalculator(const Cluster3DPCACalculator&) = delete;
  Cluster3DPCACalculator& operator=(const Cluster3DPCACalculator&) = delete;

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:
  const int _posCalcNCrystals;
  const double _logWeightDenom;
  const double _minAllowedNorm;
  
  std::unique_ptr<TPrincipal> _pca;

  void showerParameters(const reco::PFCluster&, math::XYZPoint&, 
			math::XYZVector& );

  void calculateAndSetPositionActual(reco::PFCluster&);
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  Cluster3DPCACalculator,
		  "Cluster3DPCACalculator");

void Cluster3DPCACalculator::
calculateAndSetPosition(reco::PFCluster& cluster) {
  _pca.reset(new TPrincipal(3,"D"));
  calculateAndSetPositionActual(cluster);
}

void Cluster3DPCACalculator::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    _pca.reset(new TPrincipal(3,"D"));
    calculateAndSetPositionActual(cluster);
  }
}

void Cluster3DPCACalculator::
calculateAndSetPositionActual(reco::PFCluster& cluster) {  
  if( !cluster.seed() ) {
    throw cms::Exception("ClusterWithNoSeed")
      << " Found a cluster with no seed: " << cluster;
  }  				
  double cl_energy = 0;  
  //double cl_time = 0;  
  //double cl_timeweight=0.0;
  double max_e = 0.0;  
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  reco::PFRecHitRef refseed;  
  // find the seed and max layer and also calculate time
  //Michalis : Even if we dont use timing in clustering here we should fill
  //the time information for the cluster. This should use the timing resolution(1/E)
  //so the weight should be fraction*E^2
  //calculate a simplistic depth now. The log weighted will be done
  //in different stage  
  double pcavars[3];
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
    pcavars[0] = refhit->position().x();
    pcavars[1] = refhit->position().y();
    pcavars[2] = refhit->position().z();
    for( int i = 0; i < int(rh_energy/_logWeightDenom); ++i ) {
      _pca->AddRow(pcavars);
    }
    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
    }    
  }
  cluster.setEnergy(cl_energy);
  //cluster.setTime(cl_time/cl_timeweight);
  cluster.setLayer(max_e_layer);
  // calculate the position

  _pca->MakePrincipals();
  const TVectorD& means = *(_pca->GetMeanValues());
  const TMatrixD& eigens = *(_pca->GetEigenVectors());
  /*
  std::cout << "*** Principal component analysis (PFlow) ****" << std::endl;
  std::cout << "shower average (x,y,z) = " << "(" 
	    << means[0] << ", " 
	    << means[1] << ", " 
	    << means[2] << ")" << std::endl;
  std::cout << "shower main axis (x,y,z) = " << "(" 
	    << eigens(0,0) << ", " 
	    << eigens(1,0) << ", " 
	    << eigens(2,0) << ")" << std::endl;
  */
  
  math::XYZPoint  barycenter(means[0],means[1],means[2]);
  math::XYZVector axis(eigens(0,0),eigens(1,0),eigens(2,0));

  if( axis.z()*barycenter.z() < 0.0 ) {
    axis = math::XYZVector(-eigens(0,0),-eigens(1,0),-eigens(2,0));
  }
  
  cluster.setPosition(barycenter);
  cluster.setAxis(axis);
  cluster.calculatePositionREP();
}
