#ifndef __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#define __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#include <vector>
#include <cmath>
#include <string>
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

class HGCalShowerShape{

    public:
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    const int firstLayer(const l1t::HGCalMulticluster& c3d);
	const int showerLength(const l1t::HGCalMulticluster& c3d);//in number of layers
  
    const float eMax(const l1t::HGCalMulticluster& c3d);  
  
    const float sigmaZZ(const l1t::HGCalMulticluster& c3d);

    const float sigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d);
    const float sigmaEtaEtaTot(const l1t::HGCalCluster& c2d);       
    const float sigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d);   

    const float sigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d);
    const float sigmaPhiPhiTot(const l1t::HGCalCluster& c2d);       
    const float sigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d);   

    private: 
    
    const float sigmaEtaEta(const std::vector<float> energy, const std::vector<float> eta);
    const float sigmaPhiPhi(const std::vector<float> energy, const std::vector<float> phi);   
    const float sigmaZZ(const std::vector<float> energy, const std::vector<float> z);


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
