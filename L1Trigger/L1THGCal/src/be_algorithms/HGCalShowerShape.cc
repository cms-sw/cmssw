#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "TMath.h"
#include <cmath>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>


int HGCalShowerShape::HGC_layer(const uint32_t subdet, const uint32_t layer)    const {

  int hgclayer = -1;
  if(subdet==HGCEE) hgclayer=layer;//EE
  else if(subdet==HGCHEF) hgclayer=layer+nLayersEE;//FH
  else if(subdet==HGCHEB) hgclayer=layer+nLayersEE+nLayersFH;//BH

  return hgclayer;

}





float HGCalShowerShape::sigmaEtaEta(const std::vector<float>& energy, const std::vector<float>& eta) const {

    int ntc=energy.size();
    //compute weighted eta mean
    float eta_sum=0;
    float w_sum=0; //here weight is energy
    for(int i=0;i<ntc;i++){
        eta_sum+=energy.at(i)*eta.at(i);
        w_sum+=energy.at(i);
    }
    float eta_mean=0;
    if (w_sum!=0) eta_mean=eta_sum/w_sum;
    //compute weighted eta RMS
    float deltaeta2_sum=0;
    for(int i=0;i<ntc;i++) deltaeta2_sum+=energy.at(i)*pow((eta.at(i)-eta_mean),2);
    float eta_RMS=0;
    if (w_sum!=0) eta_RMS=deltaeta2_sum/w_sum;
    float See=sqrt(eta_RMS);
    return See;
    
}


float HGCalShowerShape::sigmaPhiPhi(const std::vector<float>& energy, const std::vector<float>& phi) const {

    int ntc=energy.size();
    //compute weighted phi mean
    float phi_sum=0;
    float w_sum=0; //here weight is energy
    for(int i=0;i<ntc;i++){
        if(phi.at(i)>0) phi_sum+=energy.at(i)*phi.at(i);
        else phi_sum+=energy.at(i)*(phi.at(i)+2*TMath::Pi());
        w_sum+=energy.at(i);
    }
    float phi_mean=0;
    if (w_sum!=0) phi_mean=phi_sum/w_sum;
    if(phi_mean>TMath::Pi()) phi_mean-=2*TMath::Pi();
    //compute weighted eta RMS
    float deltaphi2_sum=0;
    for(int i=0;i<ntc;i++){
        float deltaPhi=std::abs(phi.at(i)-phi_mean);
        if (deltaPhi>TMath::Pi()) deltaPhi=2*TMath::Pi()-deltaPhi;
        deltaphi2_sum+=energy.at(i)*pow(deltaPhi,2);
    }        
    float phi_RMS=0;
    if (w_sum!=0) phi_RMS=deltaphi2_sum/w_sum;
    float Spp=sqrt(phi_RMS);
    return Spp;
    
}



float HGCalShowerShape::sigmaZZ(const std::vector<float>& energy, const std::vector<float>& z) const {

    int ntc=energy.size();
    //compute weighted eta mean
    float z_sum=0;
    float w_sum=0; //here weight is energy
    for(int i=0;i<ntc;i++){
        z_sum+=energy.at(i)*z.at(i);
        w_sum+=energy.at(i);
    }
    float z_mean=0;
    if (w_sum!=0) z_mean=z_sum/w_sum;
    //compute weighted eta RMS
    float deltaz2_sum=0;
    for(int i=0;i<ntc;i++) deltaz2_sum+=energy.at(i)*pow((z.at(i)-z_mean),2);
    float z_RMS=0;
    if (w_sum!=0) z_RMS=deltaz2_sum/w_sum;
    float Szz=sqrt(z_RMS);
    return Szz;
    
}



int HGCalShowerShape::firstLayer(const l1t::HGCalMulticluster& c3d) const {

    const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

    int firstLayer=999;

    for(const auto& clu : clustersPtrs){
      
      int layer = HGC_layer(clu->subdetId(),clu->layer());     
      if(layer<firstLayer) firstLayer=layer;

    }
    
    return firstLayer;

}



int HGCalShowerShape::lastLayer(const l1t::HGCalMulticluster& c3d) const {

    const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

    int lastLayer=-999;
    int layer=999;

    for(const auto& clu : clustersPtrs){
      
      int layer = HGC_layer(clu->subdetId(),clu->layer());     
      if(layer>lastLayer) lastLayer=layer;
      
    }

    return lastLayer;
    
}





float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const {

    const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

    std::vector<float> tc_energy ; 
    std::vector<float> tc_eta ;

    for(const auto& clu : clustersPtrs){

        const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

        for(const auto& tc : triggerCells){

            tc_energy.emplace_back(tc->energy());
            tc_eta.emplace_back(tc->eta());

        }

    }

    float SeeTot = sigmaEtaEta(tc_energy,tc_eta);

    tc_energy.clear();
    tc_eta.clear();

    return SeeTot;

}




float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const {


    const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

    std::vector<float> tc_energy ; 
    std::vector<float> tc_phi ;

    for(const auto& clu : clustersPtrs){

        const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

        for(const auto& tc : triggerCells){

            tc_energy.emplace_back(tc->energy());
            tc_phi.emplace_back(tc->phi());

        }
    }

    float SppTot = sigmaPhiPhi(tc_energy,tc_phi);

    tc_energy.clear();
    tc_phi.clear();

    return SppTot;

}



float HGCalShowerShape::sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const {

  std::map<int, vector<float> > tc_layer_energy;
  std::map<int, vector<float> > tc_layer_eta;

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

  for(const auto& clu : clustersPtrs){
    
    int layer = HGC_layer(clu->subdetId(),clu->layer());    
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

    for(const auto& tc : triggerCells){

      tc_layer_energy[layer].emplace_back(tc->energy());
      tc_layer_eta[layer].emplace_back(tc->eta());

    }

  }

  float SigmaEtaEtaMax=0;

  for(auto& tc_iter : tc_layer_energy){
    
    vector<float> energy_layer = tc_iter.second;
    vector<float> eta_layer= tc_layer_eta[tc_iter.first];

    float SigmaEtaEtaLayer = sigmaEtaEta(energy_layer,eta_layer); 
    if(SigmaEtaEtaLayer > SigmaEtaEtaMax) SigmaEtaEtaMax = SigmaEtaEtaLayer;

  }


  return SigmaEtaEtaMax;


}





float HGCalShowerShape::sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const {

  std::map<int, vector<float> > tc_layer_energy;
  std::map<int, vector<float> > tc_layer_phi;

  const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

  for(const auto& clu : clustersPtrs){
    
    int layer = HGC_layer(clu->subdetId(),clu->layer());    
    
    const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

    for(const auto& tc : triggerCells){

      tc_layer_energy[layer].emplace_back(tc->energy());
      tc_layer_phi[layer].emplace_back(tc->phi());

    }

  }

  float SigmaPhiPhiMax=0;

  for(auto& tc_iter : tc_layer_energy){
    
    vector<float> energy_layer = tc_iter.second;
    vector<float> phi_layer= tc_layer_phi[tc_iter.first];

    float SigmaPhiPhiLayer = sigmaPhiPhi(energy_layer,phi_layer); 
    if(SigmaPhiPhiLayer > SigmaPhiPhiMax) SigmaPhiPhiMax = SigmaPhiPhiLayer;

  }

  return SigmaPhiPhiMax;


}



float HGCalShowerShape::eMax(const l1t::HGCalMulticluster& c3d) const {
  
   const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

   float EMax=0;   

   for(const auto& clu : clustersPtrs){

     if(clu->energy()>EMax)
       EMax = clu->energy();

   }

   return EMax;

}






float HGCalShowerShape::sigmaZZ(const l1t::HGCalMulticluster& c3d) const {

    std::vector<float> tc_energy ; 
    std::vector<float> tc_z ;

    const edm::PtrVector<l1t::HGCalCluster>& clustersPtrs = c3d.constituents();

    for(const auto& clu : clustersPtrs){

        const edm::PtrVector<l1t::HGCalTriggerCell>& triggerCells = clu->constituents();

        for(const auto& tc : triggerCells){

            tc_energy.emplace_back(tc->energy());
            tc_z.emplace_back(tc->position().z());
       
        }
    
    }

    float Szz = sigmaZZ(tc_energy,tc_z);

    tc_energy.clear();
    tc_z.clear();

    return Szz;

}



float HGCalShowerShape::sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const {

    const edm::PtrVector<l1t::HGCalTriggerCell>& cellsPtrs = c2d.constituents();

    std::vector<float> tc_energy ; 
    std::vector<float> tc_eta ;

    for(const auto& cell : cellsPtrs){

        tc_energy.emplace_back(cell->energy());
        tc_eta.emplace_back(cell->eta());
                
    }

    float See = sigmaEtaEta(tc_energy,tc_eta);

    tc_energy.clear();
    tc_eta.clear();

    return See;

}       



float HGCalShowerShape::sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const {

    const edm::PtrVector<l1t::HGCalTriggerCell>& cellsPtrs = c2d.constituents();

    std::vector<float> tc_energy ; 
    std::vector<float> tc_phi ;

    for(const auto& cell : cellsPtrs){

            tc_energy.emplace_back(cell->energy());
            tc_phi.emplace_back(cell->phi());
                
    }

    float Spp = sigmaPhiPhi(tc_energy,tc_phi);

    tc_energy.clear();
    tc_phi.clear();

    return Spp;

}  


