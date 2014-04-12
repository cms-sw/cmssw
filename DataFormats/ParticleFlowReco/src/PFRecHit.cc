#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
using namespace reco;
using namespace std;

const unsigned    PFRecHit::nCorners_ = 4;

PFRecHit::PFRecHit() : 
  detId_(0),
  layer_(PFLayer::NONE),
  energy_(0.), 
  time_(-1.),
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
  position_(posx, posy, posz),
  axisxyz_(axisx, axisy, axisz) {

  cornersxyz_.reserve( nCorners_ );
  for(unsigned i=0; i<nCorners_; i++) { 
    cornersxyz_.push_back( position_ );    
  } 

  calculatePositionREP();

}    


PFRecHit::PFRecHit(const PFRecHit& other) :
  originalRecHit_(other.originalRecHit_),
  detId_(other.detId_), 
  layer_(other.layer_), 
  energy_(other.energy_), 
  time_(other.time_), 
  position_(other.position_), 
  positionrep_(other.positionrep_),
  axisxyz_(other.axisxyz_),
  cornersxyz_(other.cornersxyz_),
  cornersrep_(other.cornersrep_),
  neighbours_(other.neighbours_),
  neighbourInfos_(other.neighbourInfos_),
  neighbours4_(other.neighbours4_),
  neighbours8_(other.neighbours8_)
{}



PFRecHit::~PFRecHit() 
{}


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

  cornersxyz_[i] = math::XYZPoint( posx, posy, posz);
  cornersrep_[i] = REPPoint( cornersxyz_[i].Rho(),
			     cornersxyz_[i].Eta(),
			     cornersxyz_[i].Phi() );
}

void
PFRecHit::calculatePositionREP() {
  positionrep_.SetCoordinates( position_.Rho(), 
			       position_.Eta(), 
			       position_.Phi() );
  cornersrep_.resize(cornersxyz_.size());
  for( unsigned i = 0; i < cornersxyz_.size(); ++i ) {
    cornersrep_[i].SetCoordinates(cornersxyz_[i].Rho(),
				  cornersxyz_[i].Eta(),
				  cornersxyz_[i].Phi());
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
  
  neighbours_.push_back(ref);
  neighbourInfos_.push_back(bitmask);

  if (z==0) {
    neighbours8_.push_back(ref);
    //find only the 4 neighbours
    if (absx+absy==1)
      neighbours4_.push_back(ref);
  }
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
