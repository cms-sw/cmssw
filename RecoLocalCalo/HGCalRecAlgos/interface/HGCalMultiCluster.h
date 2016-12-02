#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalMultiCluster
#define RecoLocalCalo_HGCalRecAlgos_HGCalMultiCluster

#include <vector>

#include "DataFormats/EgammaReco/interface/BasicCluster.h"


class HGCalMultiCluster{

public:
  typedef std::vector<reco::BasicCluster>::const_iterator component_iterator;
  typedef std::vector<reco::BasicCluster> ClusterCollection;

  HGCalMultiCluster(){
  }
  HGCalMultiCluster(ClusterCollection &thecls) : myclusters(thecls){
  }
  HGCalMultiCluster(const reco::BasicCluster &thecl) :  myclusters(1,thecl){
  }
  void push_back(const reco::BasicCluster &b){
    myclusters.push_back(b);
  }
  unsigned int size()
  {
    return myclusters.size();
  }						
  component_iterator begin() const {
    return myclusters.begin();
  }
  component_iterator end() const {
    return myclusters.end();
  }
  double simple_z(double vz) const {
    double acc = 0.0;
    double totweight = 0.;
    for(component_iterator it = begin(); it != end(); it++){
      acc += (it->z()-vz)*it->energy()*it->size();
      totweight+=it->energy()*it->size();
    }
    return acc/totweight;
  }

  double simple_slope_x(double vz) const {
    double acc = 0.0;
    double totweight = 0.;
    for(component_iterator it = begin(); it != end(); it++){
      acc += it->x()/(it->z()-vz)*it->energy()*it->size();
      totweight+=it->energy()*it->size();
    }
    return acc/totweight;
  }
  double simple_slope_y(double vz) const {
    double acc = 0.0;
    double totweight = 0.;
    for(component_iterator it = begin(); it != end(); it++){
      acc += it->y()/(it->z()-vz)*it->energy()*it->size();
      totweight+=it->energy()*it->size();
    }
    return acc/totweight;
  }
  double simple_eta(double vz) const {
    double acc = 0.0;
    double totweight = 0.;
    for(component_iterator it = begin(); it != end(); it++){
      float point_r = sqrt(pow(it->x(),2)+pow(it->y(),2));
      float point_z = it->z()-vz;
      acc += -1. * log(tan(atan2(point_r,point_z)/2.)) * it->energy()*it->size();
      totweight+=it->energy()*it->size();
    }
    return acc/totweight;

  }

  double simple_phi() const {
    double acc = 0.0;
    int n = 0;
    for(component_iterator it = begin(); it != end(); it++){
      acc += atan2(it->y(),it->x());
      n++;
    }
    return acc/n;
  }

  double total_uncalibrated_energy() const{
    double acc = 0.0;
    for(component_iterator it = begin(); it != end(); it++){
      acc += it->energy();
    }
    return acc;
  }
  bool operator > (const HGCalMultiCluster& rhs) const { 
    return (total_uncalibrated_energy() > rhs.total_uncalibrated_energy()); 
  }
private:

  std::vector<reco::BasicCluster> myclusters;

};
#endif
