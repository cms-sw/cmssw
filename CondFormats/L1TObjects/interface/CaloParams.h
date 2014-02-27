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

#include <iostream>

namespace l1t {
  
  class CaloParams {
    
  public:

    CaloParams() {}

    ~CaloParams() {}

    // getters
    double towerLsbH() const { return towerLsbH_; }
    double towerLsbE() const { return towerLsbE_; }
    int towerNBitsH() const { return towerNBitsH_; }
    int towerNBitsE() const { return towerNBitsE_; }
    
    // setters
    void setTowerLsbH(double lsb) { towerLsbH_ = lsb; }
    void setTowerLsbE(double lsb) { towerLsbE_ = lsb; }
    void setTowerNBitsH(int n) { towerNBitsH_ = n; }
    void setTowerNBitsE(int n) { towerNBitsE_ = n; }

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }
    
  private:

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

    

  };

}// namespace
#endif
