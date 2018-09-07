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
    updateTiming_(conf.getParameter<bool>("updateTiming")),
    pca_(new TPrincipal(3,"D")){
  }
  Cluster3DPCACalculator(const Cluster3DPCACalculator&) = delete;
  Cluster3DPCACalculator& operator=(const Cluster3DPCACalculator&) = delete;

  void calculateAndSetPosition(reco::PFCluster&) override;
  void calculateAndSetPositions(reco::PFClusterCollection&) override;

 private:
  const bool updateTiming_;
  std::unique_ptr<TPrincipal> pca_;

  void showerParameters(const reco::PFCluster&, math::XYZPoint&,
			math::XYZVector& );

  void calculateAndSetPositionActual(reco::PFCluster&);
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  Cluster3DPCACalculator,
		  "Cluster3DPCACalculator");

void Cluster3DPCACalculator::
calculateAndSetPosition(reco::PFCluster& cluster) {
  pca_.reset(new TPrincipal(3,"D"));
  calculateAndSetPositionActual(cluster);
}

void Cluster3DPCACalculator::
calculateAndSetPositions(reco::PFClusterCollection& clusters) {
  for( reco::PFCluster& cluster : clusters ) {
    pca_.reset(new TPrincipal(3,"D"));
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
  double max_e = 0.0;
  double avg_time = 0.0;
  double time_norm = 0.0;
  PFLayer::Layer max_e_layer = PFLayer::NONE;
  reco::PFRecHitRef refseed;
  double pcavars[3];

  for( const reco::PFRecHitFraction& rhf : cluster.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    double rh_energy = refhit->energy();
    double rh_time = refhit->time();
    cl_energy += rh_energy * rhf.fraction();
    if( rh_time > 0.0 ) { // time == -1 means no measurement
      // all times are offset by one nanosecond in digitizer
      // remove that here so all times of flight
      // are with respect to (0,0,0)
      avg_time += (rh_time - 1.0);
      time_norm += 1.0;
    }
    if( rh_energy > max_e ) {
      max_e = rh_energy;
      max_e_layer = rhf.recHitRef()->layer();
    }
    if( refhit->detId() == cluster.seed() ) refseed = refhit;
    const double rh_fraction = rhf.fraction();
    rh_energy = refhit->energy()*rh_fraction;
    if( edm::isNotFinite(rh_energy) ) {
//temporarily changed exception to warning
//      throw cms::Exception("PFClusterAlgo")
      edm::LogWarning("PFClusterAlgo")
      <<"rechit " << refhit->detId() << " has a NaN energy... "
      << "The input of the particle flow clustering seems to be corrupted.";
      continue;
    }
    pcavars[0] = refhit->position().x();
    pcavars[1] = refhit->position().y();
    pcavars[2] = refhit->position().z();
    int nhit = int( rh_energy*100 ); // put rec_hit energy in units of 10 MeV

    for( int i = 0; i < nhit; ++i ) {
      pca_->AddRow(pcavars);
    }

  }
  cluster.setEnergy(cl_energy);
  cluster.setLayer(max_e_layer);
  // calculate the position

  pca_->MakePrincipals();
  const TVectorD& means = *(pca_->GetMeanValues());
  const TMatrixD& eigens = *(pca_->GetEigenVectors());

  math::XYZPoint  barycenter(means[0],means[1],means[2]);
  math::XYZVector axis(eigens(0,0),eigens(1,0),eigens(2,0));

  if( time_norm > 0.0 ) {
    avg_time = avg_time/time_norm;
  } else {
    avg_time = std::numeric_limits<double>::min();
  }

  if( axis.z()*barycenter.z() < 0.0 ) {
    axis = math::XYZVector(-eigens(0,0),-eigens(1,0),-eigens(2,0));
  }

  if (updateTiming_) {
    cluster.setTime(avg_time);
  }
  cluster.setPosition(barycenter);
  cluster.calculatePositionREP();

}
