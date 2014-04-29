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

#include "CondFormats/L1TObjects/interface/LUT.h"

//#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include <iostream>
#include <vector>
#include <cmath>

namespace l1t {
  
  class CaloParams {
    
  public:

    CaloParams() {}

    ~CaloParams() {}

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
    bool doTowerEncoding() const { return towerDoEncoding_; }

    void setTowerLsbH(double lsb) { towerLsbH_ = lsb; }
    void setTowerLsbE(double lsb) { towerLsbE_ = lsb; }
    void setTowerLsbSum(double lsb) { towerLsbSum_ = lsb; }
    void setTowerNBitsH(int n) { towerNBitsH_ = n; towerMaskH_ = std::pow(2,n)-1; }
    void setTowerNBitsE(int n) { towerNBitsE_ = n; towerMaskE_ = std::pow(2,n)-1; }
    void setTowerNBitsSum(int n) { towerNBitsSum_ = n; towerMaskSum_ = std::pow(2,n)-1; }
    void setTowerNBitsRatio(int n) { towerNBitsRatio_ = n; towerMaskRatio_ = std::pow(2,n)-1; }
    void setTowerEncoding(bool doit) { towerDoEncoding_ = doit; }


    // regions
    std::string regionPUSType() { return regionPUSType_; }
    std::vector<double> regionPUSParams() { return regionPUSParams_; }

    void setRegionPUSType(std::string type) { regionPUSType_ = type; }
    void setRegionPUSParams(std::vector<double> params) { regionPUSParams_ = params; }

    // EG
    double egSeedThreshold() const { return egSeedThreshold_; }
    double egNeighbourThreshold() const { return egNeighbourThreshold_; }
    double egMaxHcalEt() const { return egMaxHcalEt_; }
    double egMaxHOverE() const { return egMaxHOverE_; }
    std::string egIsoPUSType() const { return egIsoPUSType_; }
    l1t::LUT* egIsolationLUT() { return egIsolationLUT_; }

    void setEgIsoPUSType(std::string type) { egIsoPUSType_ = type; }
    void setEgIsolationLUT(LUT* lut) { egIsolationLUT_ = lut; }


    // tau
    double tauSeedThreshold() const { return tauSeedThreshold_; }
    double tauNeighbourThreshold() const { return tauNeighbourThreshold_; }


    // jets
    double jetSeedThreshold() const { return jetSeedThreshold_; }
    double jetNeighbourThreshold() const { return jetNeighbourThreshold_; }
    std::string jetPUSType() const { return jetPUSType_; }
    std::string jetCalibrationType() const { return jetCalibrationType_; }
    std::vector<double> jetCalibrationParams() { return jetCalibrationParams_; }

    void setJetSeedThreshold(double thresh) { jetSeedThreshold_ = thresh; }
    void setJetNeighbourThreshold(double thresh) { jetNeighbourThreshold_ = thresh; }
    void setJetPUSType(std::string type) { jetPUSType_ = type; }
    void setJetCalibrationType(std::string type) { jetCalibrationType_ = type; }
    void setJetCalibrationParams(std::vector<double> params) { jetCalibrationParams_ = params; }


    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }



    // redundant ?

    bool PUSubtract() const { return PUSubtract_; }           
    std::vector<double> regionSubtraction() const { return regionSubtraction_; }
    bool applyJetCalibration() const { return applyJetCalibration_; }           
    std::vector<double> jetSF() const { return jetSF_; }

    void setPUSubtract(bool pusub) { PUSubtract_ = pusub; }            
    void setregionSubtraction(std::vector<double> regsub) { regionSubtraction_ = regsub; }
    void setapplyJetCalibration(bool jetcalib) { applyJetCalibration_ = jetcalib; }            
    void setjetSF(std::vector<double> jetsf) { jetSF_ = jetsf; }
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

    // LSB of HCAL scale
    double towerLsbH_;

    // LSB of ECAL scale
    double towerLsbE_;

    // LSB of ECAL+HCAL sum scale
    double towerLsbSum_;

    // number of bits for HCAL encoding
    int towerNBitsH_;

    // number of bits for ECAL encoding
    int towerNBitsE_;

    // number of bits for ECAL+HCAL sum encoding
    int towerNBitsSum_;

    // number of bits for ECAL/HCAL ratio encoding
    int towerNBitsRatio_;

    // bitmask for storing HCAL Et in tower object
    int towerMaskH_;

    // bitmask for storing ECAL ET in tower object
    int towerMaskE_;

    // bitmask for storing ECAL+HCAL sum in tower object
    int towerMaskSum_;

    // bitmask for storing ECAL/HCAL ratio in tower object
    int towerMaskRatio_;
    
    // turn encoding on/off
    bool towerDoEncoding_;


    /* Regions */

    // PUS scheme
    std::string regionPUSType_;

    // PUS parameters
    std::vector<double> regionPUSParams_;



    /* Clustering */

    // Et threshold on EG seed tower
    double egSeedThreshold_;

    // Et threshold on EG neighbour tower(s)
    double egNeighbourThreshold_;

    // Et threshold on tau seed tower
    double tauSeedThreshold_;

    // Et threshold on tau neighbour towers
    double tauNeighbourThreshold_;
    
    // EG maximum value of HCAL Et
    double egMaxHcalEt_;

    // EG maximum value of H/E
    double egMaxHOverE_;

    // EG calibration
    // need to decide implementation
    
    // EG isolation PUS
    std::string egIsoPUSType_;

    // EG isolation LUT (indexed by eta, Et ?)
    l1t::LUT* egIsolationLUT_;

    

    /* Jets */

    // Et threshold on jet seed tower/region
    double jetSeedThreshold_;

    // Et threshold on neighbouring towers/regions
    double jetNeighbourThreshold_;

    // jet PUS scheme ("None" means no PU)
    std::string jetPUSType_;                    

    // jet calibration scheme ("None" means no JEC)
    std::string jetCalibrationType_;

    // jet calibration coefficients
    std::vector<double> jetCalibrationParams_;



    /* Sums */

    // minimum eta for EtSums (index is particular EtSum.  MET=1, ETT=2, MHT=3, HTT=4, other values reserved).
    std::vector<int> etSumEtaMin_;

    // maximum eta for EtSums (index is particular EtSum.  MET=1, ETT=2, MHT=3, HTT=4, other values reserved)
    std::vector<int> etSumEtaMax_;

    // minimum eta for EtSums (index is particular EtSum.  MET=1, ETT=2, MHT=3, HTT=4, other values reserved).  Value will be converted to int before being applied!
    std::vector<double> etSumEtThreshold_;




    /* CCLA */

    // probbaly redundant with above parameters  ?
    bool PUSubtract_;
    std::vector<double> regionSubtraction_;    //pu subtraction look up table, see region_cfi
    bool applyJetCalibration_;
    std::vector<double> jetSF_;    // jet correction table, see jet_sfi

    // these are redundant with etSumEtaMin_, etSumEtaMax_, etSumEtThreshold_ etc.
    unsigned regionETCutForHT_;
    unsigned regionETCutForMET_;
    int minGctEtaForSums_;
    int maxGctEtaForSums_;

    //redundant with L1CaloEtScale for stage 1
    // discussion needed for stage 2
    double emScale_;
    double jetScale_;
  };

}// namespace
#endif
