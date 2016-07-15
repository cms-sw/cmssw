#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <cmath>
#include <unordered_map>

#include "vdt/vdtMath.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "TPrincipal.h"

class Cluster3DPCACalculator : public PFCPositionCalculatorBase {
 public:
  Cluster3DPCACalculator(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf),    
    _pca(new TPrincipal(3,"D")){ 
  }
  Cluster3DPCACalculator(const Cluster3DPCACalculator&) = delete;
  Cluster3DPCACalculator& operator=(const Cluster3DPCACalculator&) = delete;

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:
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
  double pcavars[3];  

  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    const double rh_energy = refhit->energy();
    cl_energy += rh_energy;
    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
    }  
  }
  
  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    if( refhit->detId() == cluster.seed() ) refseed = refhit;
    const double rh_fraction = rhf.fraction();
    //const double rh_rawenergy = refhit->energy();
    const double rh_energy = refhit->energy()*rh_fraction;
    if( edm::isNotFinite(rh_energy) ) {
      throw cms::Exception("PFClusterAlgo")
	<<"rechit " << refhit->detId() << " has a NaN energy... " 
	<< "The input of the particle flow clustering seems to be corrupted.";
    }    
    pcavars[0] = refhit->position().x();
    pcavars[1] = refhit->position().y();
    pcavars[2] = refhit->position().z();     
    int nhit = int( rh_energy );

    for( int i = 0; i < nhit; ++i ) {
      _pca->AddRow(pcavars);
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
  //cluster.setAxis(axis);
  cluster.calculatePositionREP();
}
