#ifndef __L1Trigger_L1THGCal_HGCalShowerShape_h__
#define __L1Trigger_L1THGCal_HGCalShowerShape_h__
#include <vector>
#include <cmath>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalShowerShape{

    public:
    typedef math::XYZTLorentzVector LorentzVector;
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    void eventSetup(const edm::EventSetup& es) {triggerTools_.eventSetup(es);}

    int firstLayer(const l1t::HGCalMulticluster& c3d) const;
    int lastLayer(const l1t::HGCalMulticluster& c3d) const;
    int maxLayer(const l1t::HGCalMulticluster& c3d) const;
    int showerLength(const l1t::HGCalMulticluster& c3d) const {return lastLayer(c3d)-firstLayer(c3d)+1; }//in number of layers
    // Maximum number of consecutive layers in the cluster
    int coreShowerLength(const l1t::HGCalMulticluster& c3d, const HGCalTriggerGeometryBase& triggerGeometry) const;
  
    float eMax(const l1t::HGCalMulticluster& c3d) const;  
  
    float sigmaZZ(const l1t::HGCalMulticluster& c3d) const;

    float sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d) const;
    float sigmaEtaEtaTot(const l1t::HGCalCluster& c2d) const;       
    float sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d) const;   

    float sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d) const;
    float sigmaPhiPhiTot(const l1t::HGCalCluster& c2d) const;       
    float sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d) const;   

    float sigmaRRTot(const l1t::HGCalMulticluster& c3d) const;
    float sigmaRRTot(const l1t::HGCalCluster& c2d) const;       
    float sigmaRRMax(const l1t::HGCalMulticluster& c3d) const;  
    float sigmaRRMean(const l1t::HGCalMulticluster& c3d, float radius=5.) const;

    private: 
    
    float meanX(const std::vector<pair<float,float> >& energy_X_tc) const;
    float sigmaXX(const std::vector<pair<float,float> >& energy_X_tc, const float X_cluster) const;
    float sigmaPhiPhi(const std::vector<pair<float,float> >& energy_phi_tc, const float phi_cluster) const;

    HGCalTriggerTools triggerTools_;

};


#endif

