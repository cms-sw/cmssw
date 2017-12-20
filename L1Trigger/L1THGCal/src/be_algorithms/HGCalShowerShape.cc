#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "TMath.h"
#include <cmath>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>


//Compute energy-weighted mean of any variable X in the cluster

float HGCalShowerShape::meanX(const std::vector<pair<float,float> >& energy_X_tc) const {

  float Etot = 0;
  float X_sum = 0;

  for(const auto& energy_X : energy_X_tc){

    X_sum += energy_X.first*energy_X.second;
    Etot += energy_X.first;

  }

  float X_mean = 0;
  if(Etot>0) X_mean = X_sum/Etot;
  return X_mean;

}

//Compute energy-weighted RMS of any variable X in the cluster

float HGCalShowerShape::sigmaXX(const std::vector<pair<float,float> >& energy_X_tc, const float X_cluster) const {

  float Etot = 0;
  float deltaX2_sum = 0;

  for(const auto& energy_X : energy_X_tc){

    deltaX2_sum += energy_X.first * pow(energy_X.second - X_cluster,2);
    Etot += energy_X.first;

  }

  float X_MSE = 0;
  if (Etot>0) X_MSE=deltaX2_sum/Etot;
  float X_RMS=sqrt(X_MSE);
  return X_RMS;

}


//Compute energy-weighted RMS of any variable X in the cluster
//Extra care needed because of deltaPhi

float HGCalShowerShape::sigmaPhiPhi(const std::vector<pair<float,float> >& energy_phi_tc, const float phi_cluster) const {

  float Etot = 0;
  float deltaphi2_sum = 0;

  for(const auto& energy_phi : energy_phi_tc){

    deltaphi2_sum += energy_phi.first * pow(deltaPhi(energy_phi.second,phi_cluster),2);
    Etot += energy_phi.first;

  }

  float phi_MSE = 0;
  if (Etot>0) phi_MSE=deltaphi2_sum/Etot;
  float Spp=sqrt(phi_MSE);
  return Spp;

}



int HGCalShowerShape::firstLayer(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  int firstLayer=999;
  
  for(const auto& clu : clustersPtrs){
    
    int layer = triggerTools_.layerWithOffset(clu->detId());
    if(layer<firstLayer) firstLayer=layer;
    
  }
  
  return firstLayer;

}


int HGCalShowerShape::maxLayer(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  std::unordered_map<int, float> layers_pt;
  float max_pt = 0.;
  int max_layer = 0;
  for(const auto& cluster_ptr : clustersPtrs){
    unsigned layer = triggerTools_.layerWithOffset(cluster_ptr->detId());
    auto itr_insert = layers_pt.emplace(layer, 0.);
    itr_insert.first->second += cluster_ptr->pt();
    if(itr_insert.first->second>max_pt){
      max_pt = itr_insert.first->second;
      max_layer = layer;
    }
  }
  return max_layer;
}


int HGCalShowerShape::lastLayer(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  int lastLayer=-999;   
  
  for(const auto& clu : clustersPtrs){
    
    int layer = triggerTools_.layerWithOffset(clu->detId());
    if(layer>lastLayer) lastLayer=layer;
    
  }
  
  return lastLayer;
    
}

int HGCalShowerShape::coreShowerLength(const l1t::HGCalMulticluster& c3d, const HGCalTriggerGeometryBase& triggerGeometry) const
{
  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  unsigned nlayers = triggerTools_.layers(ForwardSubdetector::ForwardEmpty);
  std::vector<bool> layers(nlayers);
  for(const auto& cluster_ptr : clustersPtrs)
  {
    int layer = triggerGeometry.triggerLayer(cluster_ptr->detId());
    layers[layer-1] = true;
  }
  int length = 0;
  int maxlength = 0;
  for(bool layer : layers)
  {
    if(layer) length++;
    else length = 0;
    if(length>maxlength) maxlength = length;
  }
  return maxlength;
}


float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_eta ; 
  
  for(const auto& clu : clustersPtrs){
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();
    
    for(const auto& tc : triggerCells){
      
      tc_energy_eta.emplace_back( std::make_pair(tc->energy(),tc->eta()) );
      
    }
    
  }
  
  float SeeTot = sigmaXX(tc_energy_eta,c3d.eta());
  
  return SeeTot;
  
}




float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_phi ; 
  
  for(const auto& clu : clustersPtrs){
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();
    
    for(const auto& tc : triggerCells){
      
      tc_energy_phi.emplace_back( std::make_pair(tc->energy(),tc->phi()) );

    }

  }
  
  float SppTot = sigmaPhiPhi(tc_energy_phi,c3d.phi());
  
  return SppTot;
  
}





float HGCalShowerShape::sigmaRRTot(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_r ; 
  
  for(const auto& clu : clustersPtrs){
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();
    
    for(const auto& tc : triggerCells){
           
      float r = std::sqrt( pow(tc->position().x(),2) + pow(tc->position().y(),2) )/std::abs(tc->position().z());
      tc_energy_r.emplace_back( std::make_pair(tc->energy(),r) );

    }

  }

  float r_mean = meanX(tc_energy_r);
  float Szz = sigmaXX(tc_energy_r,r_mean);
  
  return Szz;
  
}





float HGCalShowerShape::sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const {

  std::unordered_map<int, std::vector<std::pair<float,float> > > tc_layer_energy_eta;
  std::unordered_map<int, LorentzVector> layer_LV;

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

  for(const auto& clu : clustersPtrs){
    
    unsigned layer = triggerTools_.layerWithOffset(clu->detId());
    
    layer_LV[layer] += clu->p4();

    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

    for(const auto& tc : triggerCells){

      tc_layer_energy_eta[layer].emplace_back( std::make_pair(tc->energy(),tc->eta()) );

    }

  }


  float SigmaEtaEtaMax=0;

  for(auto& tc_iter : tc_layer_energy_eta){
    
    const std::vector<std::pair<float, float> >& energy_eta_layer = tc_iter.second;
    const LorentzVector& LV_layer = layer_LV[tc_iter.first];
    float SigmaEtaEtaLayer = sigmaXX(energy_eta_layer,LV_layer.eta()); //RMS wrt layer eta, not wrt c3d eta  
    if(SigmaEtaEtaLayer > SigmaEtaEtaMax) SigmaEtaEtaMax = SigmaEtaEtaLayer;

  }


  return SigmaEtaEtaMax;


}




float HGCalShowerShape::sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const {

  std::unordered_map<int, std::vector<std::pair<float,float> > > tc_layer_energy_phi;
  std::unordered_map<int, LorentzVector> layer_LV;

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

  for(const auto& clu : clustersPtrs){
    
    unsigned layer = triggerTools_.layerWithOffset(clu->detId());
    
    layer_LV[layer] += clu->p4();

    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

    for(const auto& tc : triggerCells){

      tc_layer_energy_phi[layer].emplace_back( std::make_pair(tc->energy(),tc->phi()) );

    }

  }


  float SigmaPhiPhiMax=0;

  for(auto& tc_iter : tc_layer_energy_phi){
    
    const std::vector<std::pair<float, float> >& energy_phi_layer = tc_iter.second;
    const LorentzVector& LV_layer = layer_LV[tc_iter.first];
    float SigmaPhiPhiLayer = sigmaPhiPhi(energy_phi_layer,LV_layer.phi()); //RMS wrt layer phi, not wrt c3d phi
    if(SigmaPhiPhiLayer > SigmaPhiPhiMax) SigmaPhiPhiMax = SigmaPhiPhiLayer;

  }


  return SigmaPhiPhiMax;


}




float HGCalShowerShape::sigmaRRMax(const l1t::HGCalMulticluster& c3d) const {

  std::unordered_map<int, std::vector<std::pair<float,float> > > tc_layer_energy_r;

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

  for(const auto& clu : clustersPtrs){
    
    unsigned layer = triggerTools_.layerWithOffset(clu->detId());

    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

    for(const auto& tc : triggerCells){

      float r = std::sqrt( pow(tc->position().x(),2) + pow(tc->position().y(),2) )/std::abs(tc->position().z());
      tc_layer_energy_r[layer].emplace_back( std::make_pair(tc->energy(),r) );

    }

  }


  float SigmaRRMax=0;

  for(auto& tc_iter : tc_layer_energy_r){
    
    const std::vector<std::pair<float, float> >& energy_r_layer = tc_iter.second;
    float r_mean_layer = meanX(energy_r_layer);
    float SigmaRRLayer = sigmaXX(energy_r_layer,r_mean_layer);
    if(SigmaRRLayer > SigmaRRMax) SigmaRRMax = SigmaRRLayer;

  }


  return SigmaRRMax;


}


float HGCalShowerShape::sigmaRRMean(const l1t::HGCalMulticluster& c3d, float radius) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  // group trigger cells by layer
  std::unordered_map<int, std::vector<edm::Ptr<l1t::HGCalTriggerCell>> > layers_tcs;
  for(const auto& clu : clustersPtrs){
    unsigned layer = triggerTools_.layerWithOffset(clu->detId());
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();
    for(const auto& tc : triggerCells){
      layers_tcs[layer].emplace_back(tc);
    }
  }

  // Select trigger cells within X cm of the max TC in the layer
  std::unordered_map<int, std::vector<std::pair<float,float> > > tc_layers_energy_r;
  for(const auto& layer_tcs : layers_tcs){
    int layer = layer_tcs.first;
    edm::Ptr<l1t::HGCalTriggerCell> max_tc = layer_tcs.second.front();
    for(const auto& tc : layer_tcs.second){
      if(tc->energy()>max_tc->energy()) max_tc = tc;
    }
    for(const auto& tc : layer_tcs.second){
      double dx = tc->position().x() - max_tc->position().x();
      double dy = tc->position().y() - max_tc->position().y();
      double distance_to_max = std::sqrt(dx*dx+dy*dy);
      if(distance_to_max<radius){
        float r = std::sqrt(tc->position().x()*tc->position().x() + tc->position().y()*tc->position().y())/std::abs(tc->position().z());
        tc_layers_energy_r[layer].emplace_back( std::make_pair(tc->energy(), r));
      }
    }
  }

  // Compute srr layer by layer
  std::vector<std::pair<float,float>> layers_energy_srr2;
  for(const auto& layer_energy_r : tc_layers_energy_r){
    const auto& energies_r = layer_energy_r.second;
    float r_mean_layer = meanX(energies_r);
    float srr = sigmaXX(energies_r,r_mean_layer);
    double energy_sum = 0.;
    for(const auto& energy_r : energies_r){
      energy_sum += energy_r.first;
    }
    layers_energy_srr2.emplace_back(std::make_pair(energy_sum, srr*srr));
  }
  // Combine all layer srr
  float srr2_mean = meanX(layers_energy_srr2);
  return std::sqrt(srr2_mean);
}



float HGCalShowerShape::eMax(const l1t::HGCalMulticluster& c3d) const {
  
  std::unordered_map<int, float> layer_energy;
  
  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  for(const auto& clu : clustersPtrs){
    
    unsigned layer = triggerTools_.layerWithOffset(clu->detId());
    layer_energy[layer] += clu->energy();

  }
  
  float EMax=0;   
  
  for(const auto& layer : layer_energy){
    
    if(layer.second>EMax) EMax = layer.second;
    
  }

  return EMax;

}





float HGCalShowerShape::sigmaZZ(const l1t::HGCalMulticluster& c3d) const {

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_z ; 
  
  for(const auto& clu : clustersPtrs){
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();
    
    for(const auto& tc : triggerCells){
           
      tc_energy_z.emplace_back( std::make_pair(tc->energy(),tc->position().z()) );

    }

  }

  float z_mean = meanX(tc_energy_z);
  float Szz = sigmaXX(tc_energy_z,z_mean);
  
  return Szz;
  
}





float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const {

  const edm::PtrVector<l1t::HGCalTriggerCell>& cellsPtrs = c2d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_eta ; 
  
  for(const auto& cell : cellsPtrs){
    
    tc_energy_eta.emplace_back( std::make_pair(cell->energy(),cell->eta()) ); 
    
  }
  
  float See = sigmaXX(tc_energy_eta,c2d.eta());
  
  return See;
  
}       



float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const {

  const edm::PtrVector<l1t::HGCalTriggerCell>& cellsPtrs = c2d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_phi ; 
  
  for(const auto& cell : cellsPtrs){
    
    tc_energy_phi.emplace_back( std::make_pair(cell->energy(),cell->phi()) ); 
    
  }
  
  float Spp = sigmaPhiPhi(tc_energy_phi,c2d.phi());
  
  return Spp;
  
}      




float HGCalShowerShape::sigmaRRTot(const l1t::HGCalCluster& c2d) const {

  const edm::PtrVector<l1t::HGCalTriggerCell>& cellsPtrs = c2d.constituents();
  
  std::vector<std::pair<float,float> > tc_energy_r ; 
  
  for(const auto& cell : cellsPtrs){
    
    float r = std::sqrt( pow(cell->position().x(),2) + pow(cell->position().y(),2) )/std::abs(cell->position().z());    
    tc_energy_r.emplace_back( std::make_pair(cell->energy(),r) ); 
    
  }
  
  float r_mean = meanX(tc_energy_r);
  float Srr = sigmaXX(tc_energy_r,r_mean);
  
  return Srr;
  
}       
