// CaloParamsHelper.h
// Author: R. Alex Barbieri
//
// Wrapper class for CaloParams and Et scales

#include <iostream>

#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

#ifndef CaloParamsHelper_h
#define CaloParamsHelper_h

namespace l1t {

  class CaloParamsHelper : public CaloParams {

  public:
    // DO NOT ADD ENTRIES ANYWHERE BUT DIRECTLY BEFORE "NUM_CALOPARAMNODES"
    // DO NOT CHANGE NUMERICAL VALUES OF ANY ALREADY EXISTING FIELDS, YOU CAN ONLY EXTEND, AT THE END.
    enum { regionPUS=0,
	   egTrimming=1, egMaxHOverE=2, egCompressShapes=3, egShapeId=4, egCalibration=5, egPUS=6, egIsolation=7,
	   tauCalibration=8, tauPUS=9, tauIsolation=10,
	   jetPUS=11, jetCalibration=12,
	   hiCentrality=13, hiQ2=14,
	   tauEtToHFRingEt=15,
           tauCompress=16,
           layer1ECal=17,
           layer1HCal=18,
           layer1HF=19,
	   NUM_CALOPARAMNODES=20
    };

    CaloParamsHelper() { pnode_.resize(NUM_CALOPARAMNODES); }
    CaloParamsHelper(const CaloParams);
    ~CaloParamsHelper() {}
    


    bool isValidForStage1() {return 1; } 
    bool isValidForStage2() {return (version_ >= 2); }

    L1CaloEtScale emScale() { return emScale_; }
    void setEmScale(L1CaloEtScale emScale) { emScale_ = emScale; }
    L1CaloEtScale jetScale() { return jetScale_; }
    void setJetScale(L1CaloEtScale jetScale) { jetScale_ = jetScale; }
    L1CaloEtScale HtMissScale() {return HtMissScale_;}
    L1CaloEtScale HfRingScale() {return HfRingScale_;}
    void setHtMissScale(L1CaloEtScale HtMissScale){HtMissScale_ = HtMissScale;}
    void setHfRingScale(L1CaloEtScale HfRingScale){HfRingScale_ = HfRingScale;}

    // towers
    double towerLsbH() const { return towerp_.lsbH_; }
    double towerLsbE() const { return towerp_.lsbE_; }
    double towerLsbSum() const { return towerp_.lsbSum_; }
    int towerNBitsH() const { return towerp_.nBitsH_; }
    int towerNBitsE() const { return towerp_.nBitsE_; }
    int towerNBitsSum() const { return towerp_.nBitsSum_; }
    int towerNBitsRatio() const { return towerp_.nBitsRatio_; }
    int towerMaskE() const { return towerp_.maskE_; }
    int towerMaskH() const { return towerp_.maskH_; }
    int towerMaskSum() const { return towerp_.maskSum_; }
    int towerMaskRatio() const { return towerp_.maskRatio_; }
    bool doTowerEncoding() const { return towerp_.doEncoding_; }

    void setTowerLsbH(double lsb) { towerp_.lsbH_ = lsb; }
    void setTowerLsbE(double lsb) { towerp_.lsbE_ = lsb; }
    void setTowerLsbSum(double lsb) { towerp_.lsbSum_ = lsb; }
    void setTowerNBitsH(int n) { towerp_.nBitsH_ = n; towerp_.maskH_ = std::pow(2,n)-1; }
    void setTowerNBitsE(int n) { towerp_.nBitsE_ = n; towerp_.maskE_ = std::pow(2,n)-1; }
    void setTowerNBitsSum(int n) { towerp_.nBitsSum_ = n; towerp_.maskSum_ = std::pow(2,n)-1; }
    void setTowerNBitsRatio(int n) { towerp_.nBitsRatio_ = n; towerp_.maskRatio_ = std::pow(2,n)-1; }
    void setTowerEncoding(bool doit) { towerp_.doEncoding_ = doit; }


    // regions
    double regionLsb() const { return regionLsb_; }
    std::string regionPUSType() const { return pnode_[regionPUS].type_; }
    std::vector<double> regionPUSParams() { return pnode_[regionPUS].dparams_; }
    l1t::LUT* regionPUSLUT() {return &pnode_[regionPUS].LUT_; }

    int regionPUSValue(int PUM0, int eta)
    {
      int puSub = ceil(regionPUSParams()[18*eta+PUM0]*2);
      return puSub;
    }

    void setRegionLsb(double lsb) { regionLsb_ = lsb; }
    void setRegionPUSType(std::string type) { pnode_[regionPUS].type_ = type; }
    void setRegionPUSParams(const std::vector<double> & params) { pnode_[regionPUS].dparams_ = params; }
    void setRegionPUSLUT(const l1t::LUT & lut) { pnode_[regionPUS].LUT_ = lut; }

    // EG
    int egEtaCut() const {
      if (pnode_[egPUS].version_ ==1)
	return pnode_[egPUS].iparams_[0];
      else
	return 0;
    }
    double egLsb() const { return egp_.lsb_; }
    double egSeedThreshold() const { return egp_.seedThreshold_; }
    double egNeighbourThreshold() const { return egp_.neighbourThreshold_; }
    double egHcalThreshold() const { return egp_.hcalThreshold_; }
    l1t::LUT* egTrimmingLUT() { return &pnode_[egTrimming].LUT_; }
    double egMaxHcalEt() const { return egp_.maxHcalEt_; }
    double egMaxPtHOverE() const {return egp_.maxPtHOverE_;}
    l1t::LUT* egMaxHOverELUT() { return &pnode_[egMaxHOverE].LUT_; }
    l1t::LUT* egCompressShapesLUT() { return &pnode_[egCompressShapes].LUT_; }
    l1t::LUT* egShapeIdLUT() { return &pnode_[egShapeId].LUT_; }
    int egMinPtJetIsolation() const { return egp_.minPtJetIsolation_; }
    int egMaxPtJetIsolation() const { return egp_.maxPtJetIsolation_; }
    int egMinPtHOverEIsolation() const { return egp_.minPtHOverEIsolation_; }
    int egMaxPtHOverEIsolation() const { return egp_.maxPtHOverEIsolation_; }

    unsigned egIsoAreaNrTowersEta()const{return egp_.isoAreaNrTowersEta_;}
    unsigned egIsoAreaNrTowersPhi()const{return egp_.isoAreaNrTowersPhi_;}
    unsigned egIsoVetoNrTowersPhi()const{return egp_.isoVetoNrTowersPhi_;}
    const std::string & egPUSType() const { return pnode_[egPUS].type_; }
    const std::vector<double> & egPUSParams() const { return pnode_[egPUS].dparams_; }
    double egPUSParam(int ipar) const { return pnode_[egPUS].dparams_.at(ipar); }

    std::string egIsolationType() const { return pnode_[egIsolation].type_; }
    l1t::LUT* egIsolationLUT() { return &pnode_[egIsolation].LUT_; }
    std::string egCalibrationType() const { return pnode_[egCalibration].type_; }
    std::vector<double> egCalibrationParams() { return pnode_[egCalibration].dparams_; }
    l1t::LUT* egCalibrationLUT() { return &pnode_[egCalibration].LUT_; }

    void setEgEtaCut(int mask) {
      pnode_[egPUS].iparams_.resize(1);
      pnode_[egPUS].iparams_[0] = mask;
    }
    void setEgLsb(double lsb) { egp_.lsb_ = lsb; }
    void setEgSeedThreshold(double thresh) { egp_.seedThreshold_ = thresh; }
    void setEgNeighbourThreshold(double thresh) { egp_.neighbourThreshold_ = thresh; }
    void setEgHcalThreshold(double thresh) { egp_.hcalThreshold_ = thresh; }
    void setEgTrimmingLUT(const l1t::LUT & lut) { pnode_[egTrimming].LUT_ = lut; }
    void setEgMaxHcalEt(double cut) { egp_.maxHcalEt_ = cut; }
    void setEgMaxPtHOverE(double thresh) { egp_.maxPtHOverE_ = thresh;}
    void setEgMaxHOverELUT(const l1t::LUT & lut) { pnode_[egMaxHOverE].LUT_ = lut; }
    void setEgCompressShapesLUT(const l1t::LUT & lut) { pnode_[egCompressShapes].LUT_ = lut; }
    void setEgShapeIdLUT(const l1t::LUT & lut) { pnode_[egShapeId].LUT_ = lut; }
    void setEgMinPtJetIsolation(int cutValue) { egp_.minPtJetIsolation_ = cutValue; }
    void setEgMaxPtJetIsolation(int cutValue) { egp_.maxPtJetIsolation_ = cutValue; }
    void setEgMinPtHOverEIsolation(int cutValue) { egp_.minPtHOverEIsolation_ = cutValue; }
    void setEgMaxPtHOverEIsolation(int cutValue) { egp_.maxPtHOverEIsolation_ = cutValue; }

    void setEgIsoAreaNrTowersEta(unsigned iEgIsoAreaNrTowersEta){egp_.isoAreaNrTowersEta_=iEgIsoAreaNrTowersEta;}
    void setEgIsoAreaNrTowersPhi(unsigned iEgIsoAreaNrTowersPhi){egp_.isoAreaNrTowersPhi_=iEgIsoAreaNrTowersPhi;}
    void setEgIsoVetoNrTowersPhi(unsigned iEgIsoVetoNrTowersPhi){egp_.isoVetoNrTowersPhi_=iEgIsoVetoNrTowersPhi;}
    void setEgPUSType(std::string type) { pnode_[egPUS].type_ = type; }
    void setEgPUSParams(const std::vector<double> & params) { pnode_[egPUS].dparams_ = params; }
    void setEgIsolationType(std::string type) { pnode_[egIsolation].type_ = type; }
    void setEgIsolationLUT(const l1t::LUT & lut) { pnode_[egIsolation].LUT_ = lut; }
    void setEgCalibrationType(std::string type) { pnode_[egCalibration].type_ = type; }
    void setEgCalibrationParams(std::vector<double> params) { pnode_[egCalibration].dparams_ = params; }
    void setEgCalibrationLUT(const l1t::LUT & lut) { pnode_[egCalibration].LUT_ = lut; }

    // - recently imported:
    std::string egShapeIdType() const { return pnode_[egShapeId].type_; }
    void setEgShapeIdType(std::string type) { pnode_[egShapeId].type_ = type; }
    unsigned egShapeIdVersion() const { return pnode_[egShapeId].version_; }
    void setEgShapeIdVersion(unsigned version) { pnode_[egShapeId].version_ = version; }
    unsigned egCalibrationVersion() const { return pnode_[egCalibration].version_; }
    void setEgCalibrationVersion(unsigned version) { pnode_[egCalibration].version_ = version; }

    // tau
    int tauRegionMask() const {
      if (pnode_[tauPUS].version_ ==1)
	return pnode_[tauPUS].iparams_[0];
      else
	return 0;
    }
    double tauLsb() const { return taup_.lsb_; }
    double tauSeedThreshold() const { return taup_.seedThreshold_; }
    double tauNeighbourThreshold() const { return taup_.neighbourThreshold_; }
    double tauMaxPtTauVeto() const { return taup_.maxPtTauVeto_;}
    double tauMinPtJetIsolationB() const { return taup_.minPtJetIsolationB_;}
    double tauMaxJetIsolationB() const { return taup_.maxJetIsolationB_; }
    double tauMaxJetIsolationA() const { return taup_.maxJetIsolationA_; }
    int    isoTauEtaMin() const { return taup_.isoEtaMin_; }
    int    isoTauEtaMax() const { return taup_.isoEtaMax_; }
    std::string tauPUSType() const { return pnode_[tauPUS].type_; }
    const  std::vector<double> & tauPUSParams() const { return pnode_[tauPUS].dparams_; }
    double tauPUSParam(int ipar) const { return pnode_[tauPUS].dparams_.at(ipar); }

    l1t::LUT* tauIsolationLUT() { return &pnode_[tauIsolation].LUT_; }

    std::string tauCalibrationType() const { return pnode_[tauCalibration].type_; }
    std::vector<double> tauCalibrationParams() { return pnode_[tauCalibration].dparams_; }
    l1t::LUT* tauCalibrationLUT() { return &pnode_[tauCalibration].LUT_; }
    l1t::LUT* tauCompressLUT() { return &pnode_[tauCompress].LUT_; }

    l1t::LUT* tauEtToHFRingEtLUT() { return &pnode_[tauEtToHFRingEt].LUT_; }

    unsigned tauIsoAreaNrTowersEta()const{return taup_.isoAreaNrTowersEta_;}
    unsigned tauIsoAreaNrTowersPhi()const{return taup_.isoAreaNrTowersPhi_;}
    unsigned tauIsoVetoNrTowersPhi()const{return taup_.isoVetoNrTowersPhi_;}

    void setTauRegionMask(int mask) {
      pnode_[tauPUS].iparams_.resize(1);
      pnode_[tauPUS].iparams_[0] = mask;
    }
    void setTauLsb(double lsb) { taup_.lsb_ = lsb; }
    void setTauSeedThreshold(double thresh) { taup_.seedThreshold_ = thresh; }
    void setTauNeighbourThreshold(double thresh) { taup_.neighbourThreshold_ = thresh; }
    void setTauMaxPtTauVeto(double limit) { taup_.maxPtTauVeto_ = limit; }
    void setTauMinPtJetIsolationB(double limit) { taup_.minPtJetIsolationB_ = limit; }
    void setTauMaxJetIsolationB(double limit) { taup_.maxJetIsolationB_ = limit; }
    void setTauMaxJetIsolationA(double cutValue) { taup_.maxJetIsolationA_ = cutValue; }
    void setIsoTauEtaMin(int value) { taup_.isoEtaMin_ = value; }
    void setIsoTauEtaMax(int value) { taup_.isoEtaMax_ = value; }
    void setTauPUSType(std::string type) { pnode_[tauPUS].type_ = type; }
    void setTauIsolationLUT(const l1t::LUT & lut) { pnode_[tauIsolation].LUT_ = lut; }

    void setTauCalibrationType(std::string type) { pnode_[tauCalibration].type_ = type; }
    void setTauIsoAreaNrTowersEta(unsigned iTauIsoAreaNrTowersEta){taup_.isoAreaNrTowersEta_=iTauIsoAreaNrTowersEta;}
    void setTauIsoAreaNrTowersPhi(unsigned iTauIsoAreaNrTowersPhi){taup_.isoAreaNrTowersPhi_=iTauIsoAreaNrTowersPhi;}
    void setTauIsoVetoNrTowersPhi(unsigned iTauIsoVetoNrTowersPhi){taup_.isoVetoNrTowersPhi_=iTauIsoVetoNrTowersPhi;}

    void setTauCalibrationParams(std::vector<double> params) { pnode_[tauCalibration].dparams_ = params; }
    void setTauCalibrationLUT(const l1t::LUT & lut) { pnode_[tauCalibration].LUT_ = lut; }
    void setTauCompressLUT(const l1t::LUT & lut) { pnode_[tauCompress].LUT_ = lut; }
    void setTauPUSParams(const std::vector<double> & params) { pnode_[tauPUS].dparams_ = params; }

    void setTauEtToHFRingEtLUT(const l1t::LUT & lut) { pnode_[tauEtToHFRingEt].LUT_ = lut; }

    // jets
    double jetLsb() const { return jetp_.lsb_; }
    double jetSeedThreshold() const { return jetp_.seedThreshold_; }
    double jetNeighbourThreshold() const { return jetp_.neighbourThreshold_; }
    int jetRegionMask() const {
      if (pnode_[jetPUS].version_ ==1)
	return pnode_[jetPUS].iparams_[0];
      else
	return 0;
    }
    std::string jetPUSType() const { return pnode_[jetPUS].type_; }
    std::vector<double> jetPUSParams() { return pnode_[jetPUS].dparams_; }
    std::string jetCalibrationType() const { return pnode_[jetCalibration].type_; }
    std::vector<double> jetCalibrationParams() { return pnode_[jetCalibration].dparams_; }
    l1t::LUT* jetCalibrationLUT() { return &pnode_[jetCalibration].LUT_; }

    void setJetLsb(double lsb) { jetp_.lsb_ = lsb; }
    void setJetSeedThreshold(double thresh) { jetp_.seedThreshold_ = thresh; }
    void setJetNeighbourThreshold(double thresh) { jetp_.neighbourThreshold_ = thresh; }
    void setJetRegionMask(int mask) {
      pnode_[jetPUS].iparams_.resize(1);
      pnode_[jetPUS].iparams_[0] = mask;
    }
    void setJetPUSType(std::string type) { pnode_[jetPUS].type_ = type; }
    void setJetPUSParams(std::vector<double> params) { pnode_[jetPUS].dparams_ = params; }
    void setJetCalibrationType(std::string type) { pnode_[jetCalibration].type_ = type; }
    void setJetCalibrationParams(std::vector<double> params) { pnode_[jetCalibration].dparams_ = params; }
    void setJetCalibrationLUT(const l1t::LUT & lut) { pnode_[jetCalibration].LUT_ = lut; }

    // sums
    double etSumLsb() const { return etSumLsb_; }
    int etSumEtaMin(unsigned isum) const;
    int etSumEtaMax(unsigned isum) const;
    double etSumEtThreshold(unsigned isum) const;

    void setEtSumLsb(double lsb) { etSumLsb_ = lsb; }
    void setEtSumEtaMin(unsigned isum, int eta);
    void setEtSumEtaMax(unsigned isum, int eta);
    void setEtSumEtThreshold(unsigned isum, double thresh);

    // HI centrality
    int centralityRegionMask() const {
      if(pnode_[hiCentrality].version_ == 1)
	return pnode_[hiCentrality].iparams_[0] ;
      else
	return 0;
    }
    std::vector<int> minimumBiasThresholds() const {
      if(pnode_[hiCentrality].version_ == 1 && pnode_[hiCentrality].iparams_.size()==5) {
	std::vector<int> newVec;
	for(int i = 0; i<4; i++) {
	  newVec.push_back(pnode_[hiCentrality].iparams_.at(i+1));
	}
	return newVec;
      } else {
	std::vector<int> newVec;
	return newVec;
      }
    }
    l1t::LUT * centralityLUT() { return &pnode_[hiCentrality].LUT_; }
    void setCentralityRegionMask(int mask) {
      pnode_[hiCentrality].iparams_.resize(5);
      pnode_[hiCentrality].iparams_[0] = mask;
    }
    void setMinimumBiasThresholds(std::vector<int> thresholds) {
      pnode_[hiCentrality].iparams_.resize(5);
      for(int i = 0; i<4; i++) {
	pnode_[hiCentrality].iparams_[i+1] = thresholds.at(i);
      }
    }
    void setCentralityLUT(const l1t::LUT & lut) { pnode_[hiCentrality].LUT_ = lut; }

    // HI Q2
    l1t::LUT * q2LUT() { return &pnode_[hiQ2].LUT_; }
    void setQ2LUT(const l1t::LUT & lut) { pnode_[hiQ2].LUT_ = lut; }

    // HI parameters

    // Layer 1 LUT specification
    std::vector<double> layer1ECalScaleFactors() { return pnode_[layer1ECal].dparams_; }
    std::vector<double> layer1HCalScaleFactors() { return pnode_[layer1HCal].dparams_; }
    std::vector<double> layer1HFScaleFactors()   { return pnode_[layer1HF  ].dparams_; }
    std::vector<int> layer1ECalScaleETBins() { return pnode_[layer1ECal].iparams_; }
    std::vector<int> layer1HCalScaleETBins() { return pnode_[layer1HCal].iparams_; }
    std::vector<int> layer1HFScaleETBins()   { return pnode_[layer1HF  ].iparams_; }
    void setLayer1ECalScaleFactors(const std::vector<double> params) { pnode_[layer1ECal].dparams_ = params; }
    void setLayer1HCalScaleFactors(const std::vector<double> params) { pnode_[layer1HCal].dparams_ = params; }
    void setLayer1HFScaleFactors(const std::vector<double> params)   { pnode_[layer1HF  ].dparams_ = params; }
    void setLayer1ECalScaleETBins(const std::vector<int> params) { pnode_[layer1ECal].iparams_ = params; }
    void setLayer1HCalScaleETBins(const std::vector<int> params) { pnode_[layer1HCal].iparams_ = params; }
    void setLayer1HFScaleETBins(const std::vector<int> params)   { pnode_[layer1HF  ].iparams_ = params; }


  private:
    L1CaloEtScale emScale_;
    L1CaloEtScale jetScale_;
    L1CaloEtScale HtMissScale_;
    L1CaloEtScale HfRingScale_;
    friend std::ostream & operator<<(std::ostream &os, const CaloParamsHelper& h);
  };

  std::ostream & operator<<(std::ostream &os, const l1t::CaloParamsHelper& p);
  
}



#endif
