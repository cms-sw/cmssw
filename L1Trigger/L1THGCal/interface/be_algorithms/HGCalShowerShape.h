#ifndef __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#define __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#include <vector>
#include <cmath>
#include "TH3F.h"
#include <string>
#include "TPrincipal.h"
#include "TMatrixD.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

   
    class HGCalShowerShape{

    public:
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    //void Init2D(const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs);

    //void Init3D(const edm::PtrVector<l1t::HGCalCluster> & clustersPtr);

    void make2DshowerShape(const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs);
  
    void makeHGCalProfile(const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs);

    int firstLayer() const {return firstLayer_;} 
	int nLayers() const {return nLayers_;} 
  
    float EMax() const {return EMax_;}    
    float E0() const {return E0_;}    
    int EMaxLayer() const {return EMaxLayer_;} 
  
    float SigmaZZ() const {return SigmaZZ_;}
    float Zmean() const {return Zmean_;}          

    float SigmaEtaEta() const {return SigmaEtaEta_;}      
    float SigmaEtaEtaMax() const {return SigmaEtaEtaMax_;}    
    float SigmaEtaEta0() const {return SigmaEtaEta0_;}    
    int SigmaEtaEtaMaxLayer() const {return SigmaEtaEtaMaxLayer_;} 

    float SigmaPhiPhi() const {return SigmaPhiPhi_;}      
    float SigmaPhiPhiMax() const {return SigmaPhiPhiMax_;}    
    float SigmaPhiPhi0() const {return SigmaPhiPhi0_;}    
    int SigmaPhiPhiMaxLayer() const {return SigmaPhiPhiMaxLayer_;} 

    float dEtaMax() const {return dEtaMax_;}  
    float dPhiMax() const {return dPhiMax_;}  



    private: 
    
    float SigmaEtaEta(std::vector<float> energy, std::vector<float> eta);
    float SigmaPhiPhi(std::vector<float> energy, std::vector<float> phi);   
    float SigmaZZ(std::vector<float> energy, std::vector<float> z);
    float Zmean(std::vector<float> energy, std::vector<float> z);

    float SigmaZZ_; 
    float Zmean_; 

    float EMax_;
    float E0_;
    int EMaxLayer_;

    float SigmaEtaEta_;
    float SigmaEtaEtaMax_;
    float SigmaEtaEta0_;
    int SigmaEtaEtaMaxLayer_;

    float SigmaPhiPhi_;
    float SigmaPhiPhiMax_;
    float SigmaPhiPhi0_;
    int SigmaPhiPhiMaxLayer_;

    float dEtaMax_;
    float dPhiMax_;

    float dEta(std::vector<float> energy, std::vector<float> eta);
    float dPhi(std::vector<float> energy, std::vector<float> phi);  

    float deltaR(float eta1,float phi1, float eta2, float phi2);
   
    int firstLayer_, lastLayer_, nLayers_;

    };


#endif

// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
