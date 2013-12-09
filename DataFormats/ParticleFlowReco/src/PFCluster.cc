#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

using namespace std;
using namespace reco;


int    PFCluster::depthCorMode_ = 0;
double PFCluster::depthCorA_ = 0.89;
double PFCluster::depthCorB_ = 7.3;
double PFCluster::depthCorAp_ = 0.89;
double PFCluster::depthCorBp_ = 4.0;

const math::XYZPoint PFCluster::dummyVtx_(0,0,0);

PFCluster::PFCluster(PFLayer::Layer layer, double energy,
                     double x, double y, double z ) : 
  CaloCluster( energy, 
	       math::XYZPoint(x,y,z), 
	       PFLayer::toCaloID(layer),
	       CaloCluster::particleFlow ),
  posrep_( position_.Rho(), position_.Eta(), position_.Phi() ),
  color_(2)
{  }
  

void PFCluster::reset() {
  
  energy_ = 0;
  position_ *= 0;
  posrep_ *= 0;
  
  rechits_.clear();

  CaloCluster::reset();
  
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
  double depth = 0;
  switch(isHadron) {
  case 0: // e/gamma
    depth = corrA*(corrB + log(energy)); 
    break;
  case 1: // hadrons
    depth = corrA;
    break;
  default:
    assert(0);
    //     edm::LogError("PFCluster") << "unknown function for depth correction!"
    //                         << std::endl;
  }
  return depth;
}

void PFCluster::setLayer( PFLayer::Layer layer) {
  // cout<<"calling PFCluster::setLayer "<<layer<<endl;
  caloID_ = PFLayer::toCaloID( layer );
  // cout<<"done "<<caloID_<<endl;
}


PFLayer::Layer  PFCluster::layer() const {
  
  // cout<<"calling PFCluster::layer "<<caloID()<<" "<<PFLayer::fromCaloID( caloID() )<<endl;
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
