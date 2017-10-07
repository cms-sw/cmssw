#ifndef __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#define __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#include <vector>
#include <cmath>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

class HGCalShowerShape{

    public:
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    int firstLayer(const l1t::HGCalMulticluster& c3d) const;
    int lastLayer(const l1t::HGCalMulticluster& c3d) const;
    int showerLength(const l1t::HGCalMulticluster& c3d) const {return lastLayer(c3d)-firstLayer(c3d)+1; }//in number of layers
  
    float eMax(const l1t::HGCalMulticluster& c3d) const;  
  
    float sigmaZZ(const l1t::HGCalMulticluster& c3d) const;

    float sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const;
    float sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const;       
    float sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const;   

    float sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const;
    float sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const;       
    float sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const;   

    private: 
    
    float sigmaEtaEta(const std::vector<float>& energy, const std::vector<float>& eta) const;
    float sigmaPhiPhi(const std::vector<float>& energy, const std::vector<float>& phi) const;   
    float sigmaZZ(const std::vector<float>& energy, const std::vector<float>& z) const;

    static const int kLayersEE_=28;
    static const int kLayersFH_=12;
    static const int kLayersBH_=12;
    int HGC_layer(const uint32_t subdet, const uint32_t layer) const;


};


#endif

