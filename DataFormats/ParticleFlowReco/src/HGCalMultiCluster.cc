#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

using namespace reco;

HGCalMultiCluster::HGCalMultiCluster(ClusterCollection &thecls) :  
  PFCluster(),
  myclusters(thecls) {
  assert(myclusters.size() > 0 && "Invalid cluster collection, zero length.");
  this->setLayer(PFLayer::HGCAL);
  PFCluster::REPPoint temp;
  temp.SetRho(simple_rho());
  temp.SetEta(simple_eta(0.0));
  temp.SetPhi(simple_phi());
  this->setPosition(math::XYZPoint(temp.x(),temp.y(),temp.z()));
  this->setEnergy(total_uncalibrated_energy());
}
    
HGCalMultiCluster::HGCalMultiCluster(const edm::Ptr<reco::BasicCluster> &thecl) :  
  PFCluster() {
  this->setLayer(PFLayer::HGCAL);
  PFCluster::REPPoint temp;
  temp.SetRho(simple_rho());
  temp.SetEta(simple_eta(0.0));
  temp.SetPhi(simple_phi());
  this->setPosition(math::XYZPoint(temp.x(),temp.y(),temp.z()));
  this->setEnergy(total_uncalibrated_energy());
  myclusters.push_back(thecl);
}

double HGCalMultiCluster::simple_z(double vz) const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  double totweight = 0.;
  for( const auto& ptr : myclusters ) {
    acc += (ptr->z()-vz) * ptr->energy() * ptr->size();
    totweight+= ptr->energy() * ptr->size();
  }
  return acc/totweight;
}

double HGCalMultiCluster::simple_slope_x(double vz) const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  double totweight = 0.;
  for( const auto& ptr : myclusters ){
    const double x = ptr->x();
    const float point_z = ptr->z()-vz;
    assert(point_z != 0. && "Layer-cluster position given as reference point.");
    acc += x/(point_z) * ptr->energy() * ptr->size();
    totweight += ptr->energy() * ptr->size();
  }
  return acc/totweight;
}

double HGCalMultiCluster::simple_slope_y(double vz) const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  double totweight = 0.;
  for(const auto& ptr : myclusters ) {
    const double y = ptr->y();
    const double point_z = ptr->z()-vz;
    assert(point_z != 0. && "Layer-cluster position given as reference point.");
    acc += y/(point_z) * ptr->energy() * ptr->size();
    totweight += ptr->energy() * ptr->size();
  }
  return acc/totweight;
}

double HGCalMultiCluster::simple_eta(double vz) const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  double totweight = 0.;
  for(const auto& ptr : myclusters ) {
    const double x = ptr->x();
    const double y = ptr->y();
    const double point_r = std::sqrt(x*x+y*y);
    const double point_z = ptr->z()-vz;
    acc += -1. * std::log(std::tan(0.5*std::atan2(point_r,point_z))) * ptr->energy() * ptr->size();
    totweight += ptr->energy() * ptr->size();
  }
  return acc/totweight;  
}

double HGCalMultiCluster::simple_phi() const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  int n = 0;
  for(const auto& ptr : myclusters ) {
    const double x = ptr->x();
    const double y = ptr->y();
    assert((y != 0. || x != 0.) && "Cluster position somehow in beampipe.");
    acc += std::atan2(y,x);
    ++n;
  }
  return acc/n;
}

double HGCalMultiCluster::simple_rho() const {
  if( myclusters.size() == 0 ) return std::numeric_limits<double>::max();
  double acc = 0.0;
  int n = 0;
  for(const auto& ptr : myclusters ) {
    const double x = ptr->x();
    const double y = ptr->y();
    acc += std::sqrt(y*y + x*x);
    ++n;
  }
  return acc/n;
}
  
