///
/// \class l1t::CaloParams
///
/// Description: Placeholder for calorimeter trigger parameters
///
/// Implementation:
///    
///
/// \author: Jim Brooke
///

#ifndef CaloParams_h
#define CaloParams_h

//#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include <iostream>

namespace l1t {
  
  class CaloParams {
    
  public:

    CaloParams() {}

    ~CaloParams() {}

    // getters
/*     FirmwareVersion firmwarePP() const { return vFirmwarePP_; } */
/*     FirmwareVersion firmwareMP() const { return vFirmwareMP_; } */
    double towerLsbH() const { return towerLsbH_; }
    double towerLsbE() const { return towerLsbE_; }
    int towerNBitsH() const { return towerNBitsH_; }
    int towerNBitsE() const { return towerNBitsE_; }
    
    // setters
/*     void setFirmwarePP(FirmwareVersion v) { vFirmwarePP_ = v; } */
/*     void setFirmwareMP(FirmwareVersion v) { vFirmwareMP_ = v; } */
    void setTowerLsbH(double lsb) { towerLsbH_ = lsb; }
    void setTowerLsbE(double lsb) { towerLsbE_ = lsb; }
    void setTowerNBitsH(int n) { towerNBitsH_ = n; }
    void setTowerNBitsE(int n) { towerNBitsE_ = n; }

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }

    /* CCLA */

    void setRegionETCutForHT(unsigned etcut) { regionETCutForHT_ = etcut; }
    void setRegionETCutForMET(unsigned etcut) { regionETCutForMET_ = etcut; }
    void setMinGctEtaForSums(int eta) { minGctEtaForSums_ = eta; }
    void setMaxGctEtaForSums(int eta) { maxGctEtaForSums_ = eta; }

    void setEmScale(double scale) { emScale_ = scale; }
    void setJetScale(double scale) { jetScale_ = scale; }

    unsigned int regionETCutForHT() const { return regionETCutForHT_; }
    unsigned int regionETCutForMET() const { return regionETCutForMET_; }
    int minGctEtaForSums() const { return minGctEtaForSums_; }
    int maxGctEtaForSums() const { return maxGctEtaForSums_; }
    
    double emScale() const { return emScale_; }
    double jetScale() const { return jetScale_; }

  private:

    /* Firmware */
/*     l1t::FirmwareVersion vFirmwarePP_; */
/*     l1t::FirmwareVersion vFirmwareMP_; */

    /* Inputs */
    double towerLsbH_;
    double towerLsbE_;
    int towerNBitsH_;
    int towerNBitsE_;

    /* 	- tower masks ? */
    
    /* EG/tau clustering */
    double egSeedThreshold_;
    double egNeighbourThreshold_;
    double tauSeedThreshold_;
    double tauNeighbourThreshold_;
    
    /* EG/tau Identification */
    double egMaxH_;
    double egMaxHOverE_;
    /* 	- Shape veto (1 bool per shape) */
    /* 	- ID LUT (inputs : H, E, eta, shape) - maybe */
    
    /* EG/tau Calibration */
    /* 	-  definitely eta, maybe shape, Et */
    
    /* EG/tau Isolation */
    /* 	- LUT of isolation cut values */
    /* 	- "Granularity of # towers used for PU estimation" */
    
    /* Jets */
    double jetSeedThreshold_;
    double jetNeighbourThreshold_;
    /* 	- calibration type */
    /* 	- calibration LUT parameters (xN, can set some maximum, say 20) */
    /* 	- the calibration LUT itself */
    
    /* Sums */
    double etSumEtaMin_[10];       // minimum eta of input object (tower, region, jet)
    double etSumEtaMax_[10];       // maximum eta of input object
    double etSumEtThreshold_[10];  // threshold on input object


    /* CCLA */
    unsigned regionETCutForHT_;
    unsigned regionETCutForMET_;
    int minGctEtaForSums_;
    int maxGctEtaForSums_;

    double emScale_;
    double jetScale_;
  };

}// namespace
#endif
