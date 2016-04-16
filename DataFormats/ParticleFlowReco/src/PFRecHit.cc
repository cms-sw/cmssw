#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

using namespace reco;
using namespace std;


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


ostream& reco::operator<<(ostream& out, const reco::PFRecHit& hit) {

  if(!out) return out;

  auto const & pos = hit.positionREP();

  out<<"hit id:"<<hit.detId()
     <<" l:"<<hit.layer()
     <<" E:"<<hit.energy()
     <<" t:"<<hit.time()
     <<" rep:"<<pos.rho()<<","<<pos.eta()<<","<<pos.phi()<<"| N:";
  return out;
}
