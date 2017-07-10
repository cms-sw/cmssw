#ifndef __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#define __L1Trigger_L1THGCal_HGCALSHOWERSHAPE_h__
#include <vector>
#include <cmath>
#include "TH3F.h"
#include <string>
//#include "L1Trigger/L1THGCal/interface/LinkDef.h"
#include "TPrincipal.h"
#include "TMatrixD.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

   
    class HGCalShowerShape{

    public:
        
    HGCalShowerShape(){}

    ~HGCalShowerShape(){}

    void Init2D(/*std::vector<int> layer, std::vector<int> subdetID, std::vector<float> cl2D_energy, std::vector<int> nTC,*/ std::vector<float> tc_energy, std::vector<float> tc_eta, std::vector<float> tc_phi/*, std::vector<float> tc_x, std::vector<float> tc_y, std::vector<float> tc_z*/);

    void Init3D(std::vector<int> layer, std::vector<int> subdetID, std::vector<float> cl2D_energy, std::vector<int> nTC, std::vector<float> tc_energy, std::vector<float> tc_eta, std::vector<float> tc_phi/*, std::vector<float> tc_x, std::vector<float> tc_y, std::vector<float> tc_z*/);  

    void make2DshowerShape();
  
    void makeHGCalProfile();

    int firstLayer() const {return firstLayer_;} 
	int nLayers() const {return nLayers_;} 
  
    float EMax() const {return EMax_;}    
    float E0() const {return E0_;}    
    int EMaxLayer() const {return EMaxLayer_;} 
    std::vector<float> EnergyVector() const {return EnergyVector_;} 
  
    float SigmaZZ() const {return SigmaZZ_;}
    float Zmean() const {return Zmean_;}          

    float SigmaEtaEta() const {return SigmaEtaEta_;}      
    float SigmaEtaEtaMax() const {return SigmaEtaEtaMax_;}    
    float SigmaEtaEta0() const {return SigmaEtaEta0_;}    
    float SigmaEtaEta10() const {return SigmaEtaEta0_;}    
    int SigmaEtaEtaMaxLayer() const {return SigmaEtaEtaMaxLayer_;} 
    std::vector<float> SigmaEtaEtaVector() const {return SigmaEtaEtaVector_;}     

    float SigmaPhiPhi() const {return SigmaPhiPhi_;}      
    float SigmaPhiPhiMax() const {return SigmaPhiPhiMax_;}    
    float SigmaPhiPhi0() const {return SigmaPhiPhi0_;}    
    float SigmaPhiPhi10() const {return SigmaPhiPhi0_;}    
    int SigmaPhiPhiMaxLayer() const {return SigmaPhiPhiMaxLayer_;} 
    std::vector<float> SigmaPhiPhiVector() const {return SigmaPhiPhiVector_;}   

    float dEtaMax() const {return dEtaMax_;}  
    float dPhiMax() const {return dPhiMax_;}  

    // Compute Shower moments using principal axis of the shower (based on PCAShowerAnalysis from C.Charlot)
  /*  void showerAxisAnalysis();
    float SigmaEtaEtaTotCor() const {return SigmaEtaEtaTotCor_;}      
    float SigmaPhiPhiTotCor() const {return SigmaPhiPhiTotCor_;}    */  

  //  void make3DHistogram(std::string name, std::vector<float> x, std::vector<float> y);//work in cartesian coordinate


    private: 
    
    std::vector<int> layer_ ; // Size : ncl2D
    std::vector<int> subdetID_ ;
    std::vector<float> cl2D_energy_ ;
    std::vector<int> nTC_ ;
    std::vector<float> tc_energy_ ; // Size : ncl2D*nTCi
    std::vector<float> tc_eta_ ;
    std::vector<float> tc_phi_ ;
    std::vector<float> tc_z_ ;
    std::vector<float> tc_y_ ;
    std::vector<float> tc_x_ ;

    int ncl2D_;

    float SigmaEtaEta(std::vector<float> energy, std::vector<float> eta);
    float SigmaPhiPhi(std::vector<float> energy, std::vector<float> phi);   
    float SigmaZZ(std::vector<float> energy, std::vector<float> z);
    float Zmean(std::vector<float> energy, std::vector<float> z);

    std::vector<float> EnergyVector_; // Size : 40 (HGCal layers)
    float SigmaZZ_; 
    float Zmean_; 
    std::vector<float> SigmaEtaEtaVector_; // Size : 40 (HGCal layers)
    std::vector<float> SigmaPhiPhiVector_; // Size : 40 (HGCal layers)

    float EMax_;
    float E0_;
    int EMaxLayer_;

    float SigmaEtaEta_;
    float SigmaEtaEtaMax_;
    float SigmaEtaEta0_;
    float SigmaEtaEta10_;
    int SigmaEtaEtaMaxLayer_;

    float SigmaPhiPhi_;
    float SigmaPhiPhiMax_;
    float SigmaPhiPhi0_;
    float SigmaPhiPhi10_;
    int SigmaPhiPhiMaxLayer_;

    float dEtaMax_;
    float dPhiMax_;

    float dEta(std::vector<float> energy, std::vector<float> eta);
    float dPhi(std::vector<float> energy, std::vector<float> phi);  

    float deltaR(float eta1,float phi1, float eta2, float phi2);
   
    int firstLayer_;
    int lastLayer_;

    int nLayers_;

    TH3F *h3DShower;
    TH3F *h3DShowerFH;
    TH3F *h3DShowerEE;

    // Compute Shower moments using principal axis of the shower (based on PCAShowerAnalysis from C.Charlot)
 /*   float showerEta_;
    float showerPhi_;
    float SigmaEtaEtaTotCor_;
    float SigmaPhiPhiTotCor_;*/
    //float SigmaEtaEtaCor(std::vector<float> energy, std::vector<float> eta, float showerEta);
    //float SigmaPhiPhiCor(std::vector<float> energy, std::vector<float> phi, float showerPhi);


    //Auxiliary Containers to compute shower shapes
    std::vector<float> energy_layer_ ;
    std::vector<float> eta_layer_ ;
    std::vector<float> phi_layer_ ;
    std::vector<float> energy_ ;
    std::vector<float> eta_ ;
    std::vector<float> phi_ ;
    std::vector<float> x_ ;
    std::vector<float> y_ ;
    std::vector<float> z_ ;




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
