#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
using namespace reco;
using namespace std;

#include "vdt/vdtMath.h"
#include "Math/GenVector/etaMax.h"

const unsigned    PFRecHit::nCorners_ = 4;

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

  inline void calculateREP(const math::XYZPoint& pos, double& rho, double& eta, double& phi) {
    const double z = pos.z();
    rho = pos.Rho();    
    eta = Eta_FromRhoZ_fast<double>(rho,z);
    phi = (pos.x()==0 && pos.y()==0) ? 0 : vdt::fast_atan2(pos.y(), pos.x());
  }

}

PFRecHit::PFRecHit() : 
  detId_(0),
  layer_(PFLayer::NONE),
  energy_(0.), 
  time_(-1.),
  depth_(0),
  position_(math::XYZPoint(0.,0.,0.))
{
  
  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( position_ );    
  }
  calculatePositionREP();
}


PFRecHit::PFRecHit(unsigned detId,
                   PFLayer::Layer layer, 
                   double energy, 
                   const math::XYZPoint& position,
                   const math::XYZVector& axisxyz,
                   const vector< math::XYZPoint >& cornersxyz) : 
  detId_(detId),
  layer_(layer),
  energy_(energy), 
  time_(-1.),
  depth_(0),
  position_(position),
  axisxyz_(axisxyz),
  cornersxyz_(cornersxyz) 
{
  calculatePositionREP();
}

PFRecHit::PFRecHit(unsigned detId,
                   PFLayer::Layer layer,
                   double energy, 
                   double posx, double posy, double posz, 
                   double axisx, double axisy, double axisz) :

  detId_(detId),
  layer_(layer),
  energy_(energy), 
  time_(-1.),
  depth_(0),
  position_(posx, posy, posz),
  axisxyz_(axisx, axisy, axisz) {

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( position_ );    
  } 

  calculatePositionREP();

}    




void PFRecHit::setNWCorner( double posx, double posy, double posz ) {
  setCorner(0, posx, posy, posz);
}


void PFRecHit::setSWCorner( double posx, double posy, double posz ) {
  setCorner(1, posx, posy, posz);
}


void PFRecHit::setSECorner( double posx, double posy, double posz ) {
  setCorner(2, posx, posy, posz);
}


void PFRecHit::setNECorner( double posx, double posy, double posz ) {
  setCorner(3, posx, posy, posz);
}


void PFRecHit::setCorner( unsigned i, double posx, double posy, double posz ) {
  assert( cornersxyz_.size() == nCorners_);
  assert( i<cornersxyz_.size() );
  double rho(0), eta(0), phi(0);

  cornersxyz_[i] = math::XYZPoint( posx, posy, posz);
  const auto& corner = cornersxyz_[i];
  calculateREP(corner, rho, eta, phi);  
  cornersrep_[i] = REPPoint( rho,
			     eta,
			     phi );
}

void
PFRecHit::calculatePositionREP() {
  double rho(0), eta(0), phi(0);
  calculateREP(position_,rho,eta,phi);

  positionrep_.SetCoordinates( rho, 
			       eta, 
			       phi );
  cornersrep_.resize(cornersxyz_.size());
  for( unsigned i = 0; i < cornersxyz_.size(); ++i ) {
    const auto& corner = cornersxyz_[i];
    calculateREP(corner,rho,eta,phi);
    cornersrep_[i].SetCoordinates( rho,
                                   eta,
                                   phi );
  }
}

void PFRecHit::addNeighbour(short x,short y,short z,const PFRecHitRef& ref) {
  //bitmask interface  to accomodate more advanced naighbour finding [i.e in z as well]
  //bit 0 side for eta [0 for <=0 , 1 for >0]
  //bits 1,2,3 : abs(eta) wrt the center
  //bit 4 side for phi 
  //bits 5,6,7 : abs(phi) wrt the center
  //bit 8 side for z 
  //bits 9,10,11 : abs(z) wrt the center

  unsigned short absx = std::abs(x);
  unsigned short absy = std::abs(y);
  unsigned short absz = std::abs(z);

  unsigned short bitmask=0;


  if (x>0)
    bitmask = bitmask | 1 ;
  bitmask = bitmask | (absx << 1);
  if (y>0)
    bitmask = bitmask | (1<<4) ;
  bitmask = bitmask | (absy << 5);
  if (z>0)
    bitmask = bitmask | (1<<8) ;
  bitmask = bitmask | (absz << 9);
  
  neighbours_.push_back(ref);
  neighbourInfos_.push_back(bitmask);

  if (z==0) {
    neighbours8_.push_back(ref);
    //find only the 4 neighbours
    if (absx+absy==1)
      neighbours4_.push_back(ref);
  }
}


const PFRecHitRef PFRecHit::getNeighbour(short x,short y,short z) {
  unsigned short absx = abs(x);
  unsigned short absy = abs(y);
  unsigned short absz = abs(z);

  unsigned short bitmask=0;

  if (x>0)
    bitmask = bitmask | 1 ;
  bitmask = bitmask | (absx << 1);
  if (y>0)
    bitmask = bitmask | (1<<4) ;
  bitmask = bitmask | (absy << 5);
  if (z>0)
    bitmask = bitmask | (1<<8) ;
  bitmask = bitmask | (absz << 9);

  for (unsigned int i=0;i<neighbourInfos_.size();++i) {
    if (neighbourInfos_[i] == bitmask)
      return neighbours_[i];
  }
  return PFRecHitRef();
}


void PFRecHit::size(double& deta, double& dphi) const {

  double minphi=9999;
  double maxphi=-9999;
  double mineta=9999;
  double maxeta=-9999;
  for ( unsigned ic=0; ic<cornersxyz_.size(); ++ic ) { 
    double eta = cornersxyz_[ic].Eta();
    double phi = cornersxyz_[ic].Phi();
    
    if(phi>maxphi) maxphi=phi;
    if(phi<minphi) minphi=phi;
    if(eta>maxeta) maxeta=eta;
    if(eta<mineta) mineta=eta;    
  }

  deta = maxeta - mineta;
  dphi = maxphi - minphi;
}


ostream& reco::operator<<(ostream& out, const reco::PFRecHit& hit) {

  if(!out) return out;

  //   reco::PFRecHit& nshit = const_cast<reco::PFRecHit& >(hit);
  //   const reco::PFRecHit::REPPoint& posrep = nshit.positionREP();
  
  const  math::XYZPoint& posxyz = hit.position();

  out<<"hit id:"<<hit.detId()
     <<" l:"<<hit.layer()
     <<" E:"<<hit.energy()
     <<" t:"<<hit.time()
     <<" rep:"<<posxyz.Rho()<<","<<posxyz.Eta()<<","<<posxyz.Phi()<<"| N:";
  return out;
}
