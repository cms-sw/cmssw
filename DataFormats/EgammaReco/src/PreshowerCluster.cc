//
// $Id: PreshowerCluster.cc,v 1.8 2006/07/20 18:34:56 dbanduri Exp $
//
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace reco;

PreshowerCluster::~PreshowerCluster() { }


PreshowerCluster::PreshowerCluster(const double E, const Point& pos, 
const EcalRecHitCollection & rhits, const int plane): EcalCluster(E, pos)
{
  rhits_ = rhits;
  float X = pos.x();
  float Y = pos.y();
  float Z = pos.z();
  radius_ = sqrt(X*X + Y*Y + Z*Z);
  theta_ = pos.theta();
  eta_ = pos.eta();
  phi_ = pos.phi();
  plane_ = plane; 
  if (rhits_.size() > 0) {
     EcalRecHitCollection::iterator it;
     for (it=rhits_.begin(); it != rhits_.end(); it++) {
        usedHits_.push_back(it->id());
     }
     et_ = energy()*sin(theta());
  }
  else {
     et_ = 0.;
     eta_ = -999.;
  }

  // bc_ref_ = *BC_ref;


//   std::cout << " PreshowerCluster::PreshowerCluster, E = " << energy() << std::endl;
//   std::cout << " PreshowerCluster::PreshowerCluster, POS = " << "(" << X <<","<< Y <<","<< Z <<")"<< std::endl;
//   std::cout << " PreshowerCluster::PreshowerCluster, ETA = " << eta_ << std::endl; 

}


PreshowerCluster::PreshowerCluster(const PreshowerCluster &b) : EcalCluster( b.energy(), b.position() ) 
{
  rhits_ = b.rhits_;
  //  energy_ = b.energy_;
  et_ = b.et_;
  theta_ = b.theta_;
  radius_ = b.radius_;
  eta_ = b.eta_;
  phi_ = b.phi_;
  plane_ = b.plane_; 
}

// Comparisons
bool PreshowerCluster::operator==(const PreshowerCluster &b) const {
  double EPS = 0.000001;
  float Tdiff = fabs(b.theta() - theta());
  float Pdiff = fabs(b.phi() - phi());
  if ( (Tdiff < EPS) && (Pdiff < EPS) ) return true;
  else return false;
}

bool PreshowerCluster::operator<(const PreshowerCluster &b) const {
  return energy()*sin(theta()) < b.energy()*sin(theta()) ? true : false;
}
