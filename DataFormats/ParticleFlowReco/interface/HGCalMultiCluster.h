#ifndef DataFormats_ParticleFlowReco_HGCalMultiCluster
#define DataFormats_ParticleFlowReco_HGCalMultiCluster

#include <vector>
#include <limits>

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

namespace reco {
  class HGCalMultiCluster : public reco::PFCluster {
    
  public:
    typedef edm::PtrVector<reco::BasicCluster>::const_iterator component_iterator;
    typedef edm::PtrVector<reco::BasicCluster> ClusterCollection;
    
  HGCalMultiCluster() : PFCluster() { }

  HGCalMultiCluster(double energy,
                    double x, double y, double z,
                    ClusterCollection &thecls) :  
    PFCluster(PFLayer::HGCAL, energy, x, y, x),
    myclusters(thecls) {
      assert(myclusters.size() > 0 && "Invalid cluster collection, zero length.");
    }

  HGCalMultiCluster(ClusterCollection &thecls);
    
  HGCalMultiCluster(const edm::Ptr<reco::BasicCluster> &thecl);
    
  void push_back(const edm::Ptr<reco::BasicCluster> &b){
    myclusters.push_back(b);
  }
  
  unsigned int size() const { return myclusters.size(); }  
  component_iterator begin() const { return myclusters.begin(); }
  component_iterator end()   const { return myclusters.end(); }

  double simple_z(double vz) const;    
  double simple_slope_x(double vz) const; 
  double simple_slope_y(double vz) const; 
  
  double simple_eta(double vz) const;     
  double simple_phi() const; 
  double simple_rho() const;
  
  double total_uncalibrated_energy() const{
    double acc = 0.0;
    for(const auto& ptr : myclusters ) {
      acc += ptr->energy();
    }
    return acc;
  }
  bool operator > (const HGCalMultiCluster& rhs) const { 
    return (total_uncalibrated_energy() > rhs.total_uncalibrated_energy()); 
  }
  
  private:  
  edm::PtrVector<reco::BasicCluster> myclusters;
  
  };
}
#endif
