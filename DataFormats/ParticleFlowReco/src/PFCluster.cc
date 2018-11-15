#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "vdt/vdtMath.h"
#include "Math/GenVector/etaMax.h"

namespace {
  
  // an implementation of Eta_FromRhoZ of root libraries using vdt
  template<typename Scalar>
  inline Scalar Eta_FromRhoZ_fast(Scalar rho, Scalar z) {    
    using namespace ROOT::Math;
    // value to control Taylor expansion of sqrt
    const Scalar big_z_scaled =
      std::pow(std::numeric_limits<Scalar>::epsilon(),static_cast<Scalar>(-.25));
    if (rho > 0) {      
      Scalar z_scaled = z/rho;
      if (std::fabs(z_scaled) < big_z_scaled) {
        return vdt::fast_log(z_scaled+std::sqrt(z_scaled*z_scaled+1.0));
      } else {
        // apply correction using first order Taylor expansion of sqrt
        return  z>0 ? vdt::fast_log(2.0*z_scaled + 0.5/z_scaled) : -vdt::fast_log(-2.0*z_scaled);
      }
    }
    // case vector has rho = 0
    else if (z==0) {
      return 0;
    }
    else if (z>0) {
      return z + etaMax<Scalar>();
    }
    else {
      return z - etaMax<Scalar>();
    }    
  }
}

using namespace std;
using namespace reco;

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
std::atomic<int>    PFCluster::depthCorMode_{0};
std::atomic<double> PFCluster::depthCorA_{0.89};
std::atomic<double> PFCluster::depthCorB_{7.3};
std::atomic<double> PFCluster::depthCorAp_{0.89};
std::atomic<double> PFCluster::depthCorBp_{4.0};
#else
int    PFCluster::depthCorMode_ = 0;
double PFCluster::depthCorA_ = 0.89;
double PFCluster::depthCorB_ = 7.3;
double PFCluster::depthCorAp_ = 0.89;
double PFCluster::depthCorBp_ = 4.0;
#endif


const math::XYZPoint PFCluster::dummyVtx_(0,0,0);

PFCluster::PFCluster(PFLayer::Layer layer, double energy,
                     double x, double y, double z ) : 
  CaloCluster( energy, 
	       math::XYZPoint(x,y,z), 
	       PFLayer::toCaloID(layer),
	       CaloCluster::particleFlow ),
  posrep_( position_.Rho(), position_.Eta(), position_.Phi() ),
  time_(-99.),
  depth_(0.),
  layer_(layer),
  color_(2)
{  }
  

void PFCluster::reset() {
  
  energy_ = 0;
  position_ *= 0;
  posrep_ *= 0;
  time_=-99.;
  layer_ = PFLayer::NONE;
  rechits_.clear();

  CaloCluster::reset();
  
}

void PFCluster::resetHitsAndFractions() {

  rechits_.clear();
  hitsAndFractions_.clear();
  
}

void PFCluster::addRecHitFraction( const reco::PFRecHitFraction& frac ) {

  rechits_.push_back( frac );

  addHitAndFraction( frac.recHitRef()->detId(), 
		     frac.fraction() );
}


double PFCluster::getDepthCorrection(double energy, bool isBelowPS,
                                     bool isHadron)
{
  double corrA = depthCorA_;
  double corrB = depthCorB_;
  if (isBelowPS) {
    corrA = depthCorAp_;
    corrB = depthCorBp_;
  }
  return isHadron ? corrA : corrA*(corrB + log(energy));
}

void PFCluster::setLayer( PFLayer::Layer layer) {
  // cout<<"calling PFCluster::setLayer "<<layer<<endl;
  layer_ = layer;
  caloID_ = PFLayer::toCaloID( layer );
  // cout<<"done "<<caloID_<<endl;
}


PFLayer::Layer  PFCluster::layer() const {
  
  // cout<<"calling PFCluster::layer "<<caloID()<<" "<<PFLayer::fromCaloID( caloID() )<<endl;
  if( layer_ != PFLayer::NONE ) return layer_;
  return PFLayer::fromCaloID( caloID() );
}     


PFCluster& PFCluster::operator=(const PFCluster& other) {

  CaloCluster::operator=(other); 
  rechits_ = other.rechits_;
  energy_ = other.energy_;
  position_ = other.position_;
  posrep_ = other.posrep_;
  color_ = other.color_;

  return *this;
}


std::ostream& reco::operator<<(std::ostream& out, 
                               const PFCluster& cluster) {
  
  if(!out) return out;
  
  const math::XYZPoint&  pos = cluster.position();
  const PFCluster::REPPoint&  posrep = cluster.positionREP();
  const std::vector< reco::PFRecHitFraction >& fracs = 
    cluster.recHitFractions();
  
  out<<"PFcluster "
     <<", layer: "<<cluster.layer()
     <<"\tE = "<<cluster.energy()
     <<"\tXYZ: "
     <<pos.X()<<","<<pos.Y()<<","<<pos.Z()<<" | "
     <<"\tREP: "
     <<posrep.Rho()<<","<<posrep.Eta()<<","<<posrep.Phi()<<" | "
     <<fracs.size()<<" rechits";
  
  for(unsigned i=0; i<fracs.size(); i++) {
    // PFRecHit is not available, print the detID
    if( !fracs[i].recHitRef().isAvailable() )
      out<<cluster.printHitAndFraction(i)<<", ";
    else 
      out<<fracs[i]<<", ";
  }
  
  return out;
}
