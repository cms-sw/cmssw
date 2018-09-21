//
// T. Ruggles
// 17 Sept 2018
//
// Description: Collection representing remaining ECAL energy post-L1EG clustering
// and the associated HCAL tower. The ECAL crystal TPs "simEcalEBTriggerPrimitiveDigis"
// within a 5x5 tower region are clustered to retain this ECAL energy info for passing 
// to the GCT or further algos. The HCAL energy from the "simHcalTriggerPrimitiveDigis" 
// is not specifically used in the L1EG algo beyond H/E, so the HCAL values here
// correspond to the full, initial HCAL TP energy.


#ifndef L1CaloTower_HH
#define L1CaloTower_HH



class L1CaloTower
{
    public:
        float ecal_tower_et = 0.0;
        float hcal_tower_et = 0.0;
        int tower_iPhi = -99;
        int tower_iEta = -99;
        float tower_phi = -99;
        float tower_eta = -99;
};
  
  
// Collection of either ECAL or HCAL TPs with the Layer1 calibration constant attached, et_calibration
typedef std::vector<L1CaloTower> L1CaloTowerCollection;

#endif
