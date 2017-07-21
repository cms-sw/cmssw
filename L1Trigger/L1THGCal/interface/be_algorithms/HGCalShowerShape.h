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
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

   
    class HGCalShowerShape{

    public:
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    int firstLayer(const l1t::HGCalMulticluster& c3d);
	int nLayers(const l1t::HGCalMulticluster& c3d);
  
    float EMax(const l1t::HGCalMulticluster& c3d);  
    //float E0(const l1t::HGCalMulticluster& c3d);   
    //int EMaxLayer(const l1t::HGCalMulticluster& c3d);
  
    float SigmaZZ(const l1t::HGCalMulticluster& c3d);
    //float Zmean(const l1t::HGCalMulticluster& c3d);        

    float SigmaEtaEtaTot(const l1t::HGCalMulticluster& c3d);
    float SigmaEtaEta2D(const l1t::HGCalCluster& c2d);       
    float SigmaEtaEtaMax(const l1t::HGCalMulticluster& c3d);   
    //float SigmaEtaEta0(const l1t::HGCalMulticluster& c3d);   
    //int SigmaEtaEtaMaxLayer(const l1t::HGCalMulticluster& c3d);

    float SigmaPhiPhiTot(const l1t::HGCalMulticluster& c3d);
    float SigmaPhiPhi2D(const l1t::HGCalCluster& c2d);       
    float SigmaPhiPhiMax(const l1t::HGCalMulticluster& c3d);   
    //float SigmaPhiPhi0(const l1t::HGCalMulticluster& c3d);   
    //int SigmaPhiPhiMaxLayer(const l1t::HGCalMulticluster& c3d);


    private: 
    
    float SigmaEtaEta(std::vector<float> energy, std::vector<float> eta);
    float SigmaPhiPhi(std::vector<float> energy, std::vector<float> phi);   
    float SigmaZZ(std::vector<float> energy, std::vector<float> z);
    //float Zmean(std::vector<float> energy, std::vector<float> z);



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
