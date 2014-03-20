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

#include <cmath>

namespace l1t {
  
  class CaloParams {
    
  public:

    CaloParams() {}

    ~CaloParams() {}

    //// getters ////
    
    // towers
    double towerLsbH() const { return towerLsbH_; }
    double towerLsbE() const { return towerLsbE_; }
    double towerLsbSum() const { return towerLsbSum_; }
    int towerNBitsH() const { return towerNBitsH_; }
    int towerNBitsE() const { return towerNBitsE_; }
    int towerNBitsSum() const { return towerNBitsSum_; }
    int towerNBitsRatio() const { return towerNBitsRatio_; }
    int towerMaskE() const { return towerMaskE_; }
    int towerMaskH() const { return towerMaskH_; }
    int towerMaskSum() const { return towerMaskSum_; }
    int towerMaskRatio() const { return towerMaskRatio_; }
    bool doTowerCompression() const { return doTowerCompression_; }

    // jets
    double jetSeedThreshold() const { return jetSeedThreshold_; }

    //// setters ////

    // towers
    void setTowerLsbH(double lsb) { towerLsbH_ = lsb; }
    void setTowerLsbE(double lsb) { towerLsbE_ = lsb; }
    void setTowerLsbSum(double lsb) { towerLsbSum_ = lsb; }
    void setTowerNBitsH(int n) { towerNBitsH_ = n; towerMaskH_ = std::pow(2,n)-1; }
    void setTowerNBitsE(int n) { towerNBitsE_ = n; towerMaskE_ = std::pow(2,n)-1; }
    void setTowerNBitsSum(int n) { towerNBitsSum_ = n; towerMaskSum_ = std::pow(2,n)-1; }
    void setTowerNBitsRatio(int n) { towerNBitsRatio_ = n; towerMaskRatio_ = std::pow(2,n)-1; }
    void setTowerCompression(bool doit) { doTowerCompression_ = doit; }

    // jets
    void setJetSeedThreshold(double thresh) { jetSeedThreshold_ = thresh; }

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

    /* Towers */
    double towerLsbH_;
    double towerLsbE_;
    double towerLsbSum_;
    int towerNBitsH_;
    int towerNBitsE_;
    int towerNBitsSum_;
    int towerNBitsRatio_;
    int towerMaskH_;
    int towerMaskE_;
    int towerMaskSum_;
    int towerMaskRatio_;
    bool doTowerCompression_;

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
