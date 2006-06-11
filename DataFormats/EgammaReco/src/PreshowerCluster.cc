//
// $Id: $
//
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace reco;

PreshowerCluster::PreshowerCluster() {
  energy  = 0;
  euncorrected = 0;
  et = 0;
  radius = 0;
  theta = 0;
  eta = 0;
  phi = 0;
  layer = 0;
}

PreshowerCluster::~PreshowerCluster() { }

PreshowerCluster::PreshowerCluster(const Point& pos, const EcalRecHitCollection & rhits_, int layer_) //:
{  
  rhits = rhits_;
  float x = pos.x();
  float y = pos.y();
  float z = pos.z();
  radius = sqrt(x*x+y*y+z*z);
  theta = pos.theta();
  eta = pos.eta();
  phi = pos.phi();
  layer = layer_; 

  if (rhits.size()>0) {
    EcalRecHitCollection::iterator it;
    for (it=rhits.begin(); it != rhits.end(); it++) {
      energy += it->energy();	  
    }
    // cluster calibration?
    euncorrected = energy;
    et = energy*sin(Theta());
  }
  else {
    euncorrected = 0.;
    et = 0.;
    eta = -999.;
  }

}

PreshowerCluster::PreshowerCluster(const PreshowerCluster &b) {
  rhits = b.rhits;
  energy = b.energy;
  et = b.et;
  theta = b.theta;
  radius = b.radius;
  eta = b.eta;
  phi = b.phi;
  layer = b.layer; 
}


// Comparisons

bool PreshowerCluster::operator==(const PreshowerCluster &b) const {
  double EPS = 0.000001;
  float Tdiff = fabs(b.Theta() - Theta());
  float Pdiff = fabs(b.Phi() - Phi());
  if ( (Tdiff < EPS) && (Pdiff < EPS) ) return true;
  else return false;
}

bool PreshowerCluster::operator<(const PreshowerCluster &b) const {
  return energy*sin(Theta()) < b.energy*sin(Theta()) ? true : false;
}
