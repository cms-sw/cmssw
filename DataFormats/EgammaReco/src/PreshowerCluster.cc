//
// $Id: PreshowerCluster.cc,v 1.17 2013/04/22 22:53:03 wmtan Exp $
//
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"

using namespace reco;

PreshowerCluster::~PreshowerCluster() { }


PreshowerCluster::PreshowerCluster(const double E, const Point& pos,
				   const std::vector< std::pair<DetId, float> >& hitsAndFractions,
				   const int plane) : CaloCluster(E, pos)
{
  hitsAndFractions_ = hitsAndFractions;
  plane_ = plane;

//   std::cout << " PreshowerCluster::PreshowerCluster, E = " << energy() << std::endl;
//   std::cout << " PreshowerCluster::PreshowerCluster, POS = " << "(" << x() <<","<< y() <<","<< z() <<")"<< std::endl;
//   std::cout << " PreshowerCluster::PreshowerCluster, ETA = " << eta() << std::endl; 

}


PreshowerCluster::PreshowerCluster(const PreshowerCluster &b) : CaloCluster( b.energy(), b.position() ) 
{
  hitsAndFractions_ = b.hitsAndFractions_;
  plane_ = b.plane_; 
  bc_ref_=b.bc_ref_;
}


// Comparisons

bool PreshowerCluster::operator==(const PreshowerCluster &b) const {
  double EPS = 0.000001;
  float Tdiff = fabs(b.position().theta() - position().theta());
  float Pdiff = fabs(b.phi() - phi());
  if ( (Tdiff < EPS) && (Pdiff < EPS) ) return true;
  else return false;
}

bool PreshowerCluster::operator<(const PreshowerCluster &b) const {
  return energy()*sin(position().theta()) < b.energy()*sin(position().theta()) ? true : false;
}
