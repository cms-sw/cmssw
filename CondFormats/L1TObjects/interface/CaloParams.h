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

#include <memory>
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
    double regionLsb() const { return regionLsb_; }
    std::string regionPUSType() const { return regionPUSType_; }
    std::vector<double> regionPUSParams() { return regionPUSParams_; }

    void setRegionLsb(double lsb) { regionLsb_ = lsb; }
    void setRegionPUSType(std::string type) { regionPUSType_ = type; }
    void setRegionPUSParams(std::vector<double> params) { regionPUSParams_ = params; }


    // EG
    double egLsb() const { return egLsb_; }
    double egSeedThreshold() const { return egSeedThreshold_; }
    double egNeighbourThreshold() const { return egNeighbourThreshold_; }
    double egHcalThreshold() const { return egHcalThreshold_; }
    double egMaxHcalEt() const { return egMaxHcalEt_; }
    double egEtToRemoveHECut() const {return egEtToRemoveHECut_;}
    l1t::LUT* egMaxHOverELUT() { return egMaxHOverELUT_.get(); }
    l1t::LUT* egShapeIdLUT() { return egShapeIdLUT_.get(); }
    double egRelativeJetIsolationBarrelCut() const { return egRelativeJetIsolationBarrelCut_; }
    double egRelativeJetIsolationEndcapCut() const { return egRelativeJetIsolationEndcapCut_; }
    unsigned egIsoAreaNrTowersEta()const{return egIsoAreaNrTowersEta_;}
    unsigned egIsoAreaNrTowersPhi()const{return egIsoAreaNrTowersPhi_;}
    unsigned egIsoVetoNrTowersPhi()const{return egIsoVetoNrTowersPhi_;}
    unsigned egIsoPUEstTowerGranularity()const{return egIsoPUEstTowerGranularity_;}
    unsigned egIsoMaxEtaAbsForTowerSum()const{return egIsoMaxEtaAbsForTowerSum_;}
    unsigned egIsoMaxEtaAbsForIsoSum()const{return egIsoMaxEtaAbsForIsoSum_;}
    std::string egIsoPUSType() const { return egIsoPUSType_; }
    l1t::LUT* egIsolationLUT() { return egIsolationLUT_.get(); }
    std::string egCalibrationType() const { return egCalibrationType_; }
    std::vector<double> egCalibrationParams() { return egCalibrationParams_; }
    l1t::LUT* egCalibrationLUT() { return egCalibrationLUT_.get(); }

    void setEgLsb(double lsb) { egLsb_ = lsb; }
    void setEgSeedThreshold(double thresh) { egSeedThreshold_ = thresh; }
    void setEgNeighbourThreshold(double thresh) { egNeighbourThreshold_ = thresh; }
    void setEgHcalThreshold(double thresh) { egHcalThreshold_ = thresh; }
    void setEgMaxHcalEt(double cut) { egMaxHcalEt_ = cut; }
    void setEgEtToRemoveHECut(double thresh) { egEtToRemoveHECut_ = thresh;}
    void setEgMaxHOverELUT(std::shared_ptr<LUT> lut) { egMaxHOverELUT_ = lut; }
    void setEgShapeIdLUT(std::shared_ptr<LUT> lut) { egShapeIdLUT_ = lut; }
    void setEgRelativeJetIsolationBarrelCut(double cutValue) { egRelativeJetIsolationBarrelCut_ = cutValue; }
    void setEgRelativeJetIsolationEndcapCut(double cutValue) { egRelativeJetIsolationEndcapCut_ = cutValue; }

    void setEgIsoAreaNrTowersEta(unsigned iEgIsoAreaNrTowersEta){egIsoAreaNrTowersEta_=iEgIsoAreaNrTowersEta;}
    void setEgIsoAreaNrTowersPhi(unsigned iEgIsoAreaNrTowersPhi){egIsoAreaNrTowersPhi_=iEgIsoAreaNrTowersPhi;}
    void setEgIsoVetoNrTowersPhi(unsigned iEgIsoVetoNrTowersPhi){egIsoVetoNrTowersPhi_=iEgIsoVetoNrTowersPhi;}
    void setEgIsoPUEstTowerGranularity(unsigned iEgIsoPUEstTowerGranularity){egIsoPUEstTowerGranularity_=iEgIsoPUEstTowerGranularity;}
    void setEgIsoMaxEtaAbsForTowerSum(unsigned iEgIsoMaxEtaAbsForTowerSum){egIsoMaxEtaAbsForTowerSum_=iEgIsoMaxEtaAbsForTowerSum;}
    void setEgIsoMaxEtaAbsForIsoSum(unsigned iEgIsoMaxEtaAbsForIsoSum){egIsoMaxEtaAbsForIsoSum_=iEgIsoMaxEtaAbsForIsoSum;}
    void setEgIsoPUSType(std::string type) { egIsoPUSType_ = type; }
    void setEgIsolationLUT(std::shared_ptr<LUT> lut) { egIsolationLUT_ = lut; }
    void setEgCalibrationType(std::string type) { egCalibrationType_ = type; }
    void setEgCalibrationParams(std::vector<double> params) { egCalibrationParams_ = params; }
    void setEgCalibrationLUT(std::shared_ptr<LUT> lut) { egCalibrationLUT_ = lut; }


    // tau
    double tauLsb() const { return tauLsb_; }
    double tauSeedThreshold() const { return tauSeedThreshold_; }
    double tauNeighbourThreshold() const { return tauNeighbourThreshold_; }
    double switchOffTauVeto() const { return switchOffTauVeto_;}
    double switchOffTauIso() const { return switchOffTauIso_;}
    double tauRelativeJetIsolationLimit() const { return tauRelativeJetIsolationLimit_; }
    double tauRelativeJetIsolationCut() const { return tauRelativeJetIsolationCut_; }
    std::string tauIsoPUSType() const { return tauIsoPUSType_; }
    l1t::LUT* tauIsolationLUT() { return tauIsolationLUT_.get(); }
    std::string tauCalibrationType() const { return tauCalibrationType_; }
    std::vector<double> tauCalibrationParams() { return tauCalibrationParams_; }
    l1t::LUT* tauCalibrationLUTBarrelA() { return tauCalibrationLUTBarrelA_.get(); }
    l1t::LUT* tauCalibrationLUTBarrelB() { return tauCalibrationLUTBarrelB_.get(); }
    l1t::LUT* tauCalibrationLUTBarrelC() { return tauCalibrationLUTBarrelC_.get(); }
    l1t::LUT* tauCalibrationLUTEndcapsA() { return tauCalibrationLUTEndcapsA_.get(); }
    l1t::LUT* tauCalibrationLUTEndcapsB() { return tauCalibrationLUTEndcapsB_.get(); }
    l1t::LUT* tauCalibrationLUTEndcapsC() { return tauCalibrationLUTEndcapsC_.get(); }
    l1t::LUT* tauCalibrationLUTEta() { return tauCalibrationLUTEta_.get(); }

    void setTauLsb(double lsb) { tauLsb_ = lsb; }
    void setTauSeedThreshold(double thresh) { tauSeedThreshold_ = thresh; }
    void setTauNeighbourThreshold(double thresh) { tauNeighbourThreshold_ = thresh; }
    void setSwitchOffTauVeto(double limit) { switchOffTauVeto_ = limit; }
    void setSwitchOffTauIso(double limit) { switchOffTauIso_ = limit; }
    void setTauRelativeJetIsolationLimit(double limit) { tauRelativeJetIsolationLimit_ = limit; }
    void setTauRelativeJetIsolationCut(double cutValue) { tauRelativeJetIsolationCut_ = cutValue; }
    void setTauIsoPUSType(std::string type) { tauIsoPUSType_ = type; }
    void setTauIsolationLUT(std::shared_ptr<LUT> lut) { tauIsolationLUT_ = lut; }
    void setTauCalibrationType(std::string type) { tauCalibrationType_ = type; }
    void setTauCalibrationParams(std::vector<double> params) { tauCalibrationParams_ = params; }
    void setTauCalibrationLUTBarrelA(std::shared_ptr<LUT> lut) { tauCalibrationLUTBarrelA_ = lut; }
    void setTauCalibrationLUTBarrelB(std::shared_ptr<LUT> lut) { tauCalibrationLUTBarrelB_ = lut; }
    void setTauCalibrationLUTBarrelC(std::shared_ptr<LUT> lut) { tauCalibrationLUTBarrelC_ = lut; }
    void setTauCalibrationLUTEndcapsA(std::shared_ptr<LUT> lut) { tauCalibrationLUTEndcapsA_ = lut; }
    void setTauCalibrationLUTEndcapsB(std::shared_ptr<LUT> lut) { tauCalibrationLUTEndcapsB_ = lut; }
    void setTauCalibrationLUTEndcapsC(std::shared_ptr<LUT> lut) { tauCalibrationLUTEndcapsC_ = lut; }
    void setTauCalibrationLUTEta(std::shared_ptr<LUT> lut) { tauCalibrationLUTEta_ = lut; }

    // jets
    double jetLsb() const { return jetLsb_; }
    double jetSeedThreshold() const { return jetSeedThreshold_; }
    double jetNeighbourThreshold() const { return jetNeighbourThreshold_; }
    std::string jetPUSType() const { return jetPUSType_; }
    std::vector<double> jetPUSParams() { return jetPUSParams_; }
    std::string jetCalibrationType() const { return jetCalibrationType_; }
    std::vector<double> jetCalibrationParams() { return jetCalibrationParams_; }

    void setJetLsb(double lsb) { jetLsb_ = lsb; }
    void setJetSeedThreshold(double thresh) { jetSeedThreshold_ = thresh; }
    void setJetNeighbourThreshold(double thresh) { jetNeighbourThreshold_ = thresh; }
    void setJetPUSType(std::string type) { jetPUSType_ = type; }
    void setJetPUSParams(std::vector<double> params) { jetPUSParams_ = params; }
    void setJetCalibrationType(std::string type) { jetCalibrationType_ = type; }
    void setJetCalibrationParams(std::vector<double> params) { jetCalibrationParams_ = params; }


    // sums
    double etSumLsb() const { return etSumLsb_; }
    int etSumEtaMin(unsigned isum) const;
    int etSumEtaMax(unsigned isum) const;
    double etSumEtThreshold(unsigned isum) const;

    void setEtSumLsb(double lsb) { etSumLsb_ = lsb; }
    void setEtSumEtaMin(unsigned isum, int eta);
    void setEtSumEtaMax(unsigned isum, int eta);
    void setEtSumEtThreshold(unsigned isum, double thresh);

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }


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

    // Region LSB
    double regionLsb_;

    // PUS scheme
    std::string regionPUSType_;

    // PUS parameters
    std::vector<double> regionPUSParams_;



    /* EG */

    // EG LSB
    double egLsb_;

    // Et threshold on EG seed tower
    double egSeedThreshold_;

    // Et threshold on EG neighbour tower(s)
    double egNeighbourThreshold_;

    // Et threshold on HCAL for H/E computation
    double egHcalThreshold_;

    // EG maximum value of HCAL Et
    double egMaxHcalEt_;

    // Et threshold to remove the H/E cut from the EGammas
    double egEtToRemoveHECut_;

    // EG maximum values of H/E (indexed by |ieta|, ??)
    std::shared_ptr<l1t::LUT> egMaxHOverELUT_;

    // Shape identification bits (indexed by |ieta|, shape)
    std::shared_ptr<l1t::LUT> egShapeIdLUT_;

    // Relative jet isolation cut for EG in the barrel (Stage1Layer2)
    double egRelativeJetIsolationBarrelCut_;

    // Relative jet isolation cut for EG in the endcap (Stage1Layer2)
    double egRelativeJetIsolationEndcapCut_;

    // isolation area in eta is seed tower +/- <=egIsoAreaNrTowersPhi
    unsigned egIsoAreaNrTowersEta_;

    // isolation area in phi is seed tower +/- <=egIsoAreaNrTowersPhi
    unsigned egIsoAreaNrTowersPhi_;

    // veto region is seed tower +/- <=egIsoVetoNrTowersPhi
    unsigned egIsoVetoNrTowersPhi_;

    // for # towers based PU estimator, estimator is #towers/egIsoPUEstTowerGranularity_
    unsigned egIsoPUEstTowerGranularity_;

    // eta range over which # towers is estimated
    unsigned egIsoMaxEtaAbsForTowerSum_;

    // max abs eta for which a tower is included in the isolation sum
    unsigned egIsoMaxEtaAbsForIsoSum_;

    // EG calibration
    std::string egCalibrationType_;

    // EG calibration coefficients
    std::vector<double> egCalibrationParams_;
    std::shared_ptr<l1t::LUT> egCalibrationLUT_;

    // EG isolation PUS
    std::string egIsoPUSType_;

    // EG isolation LUT (indexed by eta, Et ?)
    std::shared_ptr<l1t::LUT> egIsolationLUT_;




    /* Tau */

    // Tau LSB
    double tauLsb_;

    // Et threshold on tau seed tower
    double tauSeedThreshold_;

    // Et threshold on tau neighbour towers
    double tauNeighbourThreshold_;

    // Et limit when to switch off tau veto requirement
    double switchOffTauVeto_;

    // Et limit when to switch off tau isolation requirement
    double switchOffTauIso_;

    // Et jet isolation limit for Taus (Stage1Layer2)
    double tauRelativeJetIsolationLimit_;

    // Relative jet isolation cut for Taus (Stage1Layer2)
    double tauRelativeJetIsolationCut_;

    // Tau isolation PUS
    std::string tauIsoPUSType_;

    // Tau isolation LUT (indexed by eta, Et ?)
     std::shared_ptr<l1t::LUT> tauIsolationLUT_;

    // Tau calibration
    std::string tauCalibrationType_;

    // Tau calibration coefficients
    std::vector<double> tauCalibrationParams_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTBarrelA_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTBarrelB_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTBarrelC_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTEndcapsA_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTEndcapsB_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTEndcapsC_;
    std::shared_ptr<l1t::LUT> tauCalibrationLUTEta_;



    /* Jets */

    // Jet LSB
    double jetLsb_;

    // Et threshold on jet seed tower/region
    double jetSeedThreshold_;

    // Et threshold on neighbouring towers/regions
    double jetNeighbourThreshold_;

    // jet PUS scheme ("None" means no PU)
    std::string jetPUSType_;

    // jet PU params
    std::vector<double> jetPUSParams_;

    // jet calibration scheme ("None" means no JEC)
    std::string jetCalibrationType_;

    // jet calibration coefficients
    std::vector<double> jetCalibrationParams_;




    /* Sums */

    // EtSum LSB
    double etSumLsb_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMin_;

    // maximum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMax_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<double> etSumEtThreshold_;



  };

}// namespace
#endif
