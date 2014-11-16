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
    std::string regionPUSType() const { return regionPUS_.type_; }
    std::vector<double> regionPUSParams() { return regionPUS_.dparams_; }

    void setRegionLsb(double lsb) { regionLsb_ = lsb; }
    void setRegionPUSType(std::string type) { regionPUS_.type_ = type; }
    void setRegionPUSParams(std::vector<double> params) { regionPUS_.dparams_ = params; }


    // EG
    double egLsb() const { return egLsb_; }
    double egSeedThreshold() const { return egSeedThreshold_; }
    double egNeighbourThreshold() const { return egNeighbourThreshold_; }
    double egHcalThreshold() const { return egHcalThreshold_; }
    l1t::LUT* egTrimmingLUT() { return egTrimming_.LUT_.get(); }
    double egMaxHcalEt() const { return egMaxHcalEt_; }
    double egMaxPtHOverE() const {return egMaxPtHOverE_;}
    l1t::LUT* egMaxHOverELUT() { return egMaxHOverE_.LUT_.get(); }
    l1t::LUT* egCompressShapesLUT() { return egCompressShapes_.LUT_.get(); }
    l1t::LUT* egShapeIdLUT() { return egShapeId_.LUT_.get(); }
    int egMinPtJetIsolation() const { return egMinPtJetIsolation_; }
    int egMaxPtJetIsolation() const { return egMaxPtJetIsolation_; }
    int egMinPtHOverEIsolation() const { return egMinPtHOverEIsolation_; }
    int egMaxPtHOverEIsolation() const { return egMaxPtHOverEIsolation_; }

    unsigned egIsoAreaNrTowersEta()const{return egIsoAreaNrTowersEta_;}
    unsigned egIsoAreaNrTowersPhi()const{return egIsoAreaNrTowersPhi_;}
    unsigned egIsoVetoNrTowersPhi()const{return egIsoVetoNrTowersPhi_;}
    //unsigned egIsoPUEstTowerGranularity()const{return egIsoPUEstTowerGranularity_;}
    //unsigned egIsoMaxEtaAbsForTowerSum()const{return egIsoMaxEtaAbsForTowerSum_;}
    //unsigned egIsoMaxEtaAbsForIsoSum()const{return egIsoMaxEtaAbsForIsoSum_;}
    
    const std::string & egPUSType() const { return egPUS_.type_; }    
    const std::vector<double> & egPUSParams() const { return egPUS_.dparams_; }
    double egPUSParam(int ipar) const { return egPUS_.dparams_.at(ipar); }
    
    
    
    l1t::LUT* egIsolationLUT() { return egIsolation_.LUT_.get(); }
    std::string egCalibrationType() const { return egCalibration_.type_; }
    std::vector<double> egCalibrationParams() { return egCalibration_.dparams_; }
    l1t::LUT* egCalibrationLUT() { return egCalibration_.LUT_.get(); }

    void setEgLsb(double lsb) { egLsb_ = lsb; }
    void setEgSeedThreshold(double thresh) { egSeedThreshold_ = thresh; }
    void setEgNeighbourThreshold(double thresh) { egNeighbourThreshold_ = thresh; }
    void setEgHcalThreshold(double thresh) { egHcalThreshold_ = thresh; }
    void setEgTrimmingLUT(std::shared_ptr<LUT> lut) { egTrimming_.LUT_ = lut; }
    void setEgMaxHcalEt(double cut) { egMaxHcalEt_ = cut; }
    void setEgMaxPtHOverE(double thresh) { egMaxPtHOverE_ = thresh;}
    void setEgMaxHOverELUT(std::shared_ptr<LUT> lut) { egMaxHOverE_.LUT_ = lut; }
    void setEgCompressShapesLUT(std::shared_ptr<LUT> lut) { egCompressShapes_.LUT_ = lut; }
    void setEgShapeIdLUT(std::shared_ptr<LUT> lut) { egShapeId_.LUT_ = lut; }
    void setEgMinPtJetIsolation(int cutValue) { egMinPtJetIsolation_ = cutValue; }
    void setEgMaxPtJetIsolation(int cutValue) { egMaxPtJetIsolation_ = cutValue; }
    void setEgMinPtHOverEIsolation(int cutValue) { egMinPtHOverEIsolation_ = cutValue; }
    void setEgMaxPtHOverEIsolation(int cutValue) { egMaxPtHOverEIsolation_ = cutValue; }

    void setEgIsoAreaNrTowersEta(unsigned iEgIsoAreaNrTowersEta){egIsoAreaNrTowersEta_=iEgIsoAreaNrTowersEta;}
    void setEgIsoAreaNrTowersPhi(unsigned iEgIsoAreaNrTowersPhi){egIsoAreaNrTowersPhi_=iEgIsoAreaNrTowersPhi;}
    void setEgIsoVetoNrTowersPhi(unsigned iEgIsoVetoNrTowersPhi){egIsoVetoNrTowersPhi_=iEgIsoVetoNrTowersPhi;}
    //void setEgIsoPUEstTowerGranularity(unsigned iEgIsoPUEstTowerGranularity){egIsoPUEstTowerGranularity_=iEgIsoPUEstTowerGranularity;}
    //void setEgIsoMaxEtaAbsForTowerSum(unsigned iEgIsoMaxEtaAbsForTowerSum){egIsoMaxEtaAbsForTowerSum_=iEgIsoMaxEtaAbsForTowerSum;}
    //void setEgIsoMaxEtaAbsForIsoSum(unsigned iEgIsoMaxEtaAbsForIsoSum){egIsoMaxEtaAbsForIsoSum_=iEgIsoMaxEtaAbsForIsoSum;}
    void setEgPUSType(std::string type) { egPUS_.type_ = type; }
    void setEgPUSParams(const std::vector<double> & params) { egPUS_.dparams_ = params; }
    void setEgIsolationLUT(std::shared_ptr<LUT> lut) { egIsolation_.LUT_ = lut; }
    void setEgCalibrationType(std::string type) { egCalibration_.type_ = type; }
    void setEgCalibrationParams(std::vector<double> params) { egCalibration_.dparams_ = params; }
    void setEgCalibrationLUT(std::shared_ptr<LUT> lut) { egCalibration_.LUT_ = lut; }


    // tau
    double tauLsb() const { return tauLsb_; }
    double tauSeedThreshold() const { return tauSeedThreshold_; }
    double tauNeighbourThreshold() const { return tauNeighbourThreshold_; }
    double tauMaxPtTauVeto() const { return tauMaxPtTauVeto_;}
    double tauMinPtJetIsolationB() const { return tauMinPtJetIsolationB_;}
    double tauMaxJetIsolationB() const { return tauMaxJetIsolationB_; }
    double tauMaxJetIsolationA() const { return tauMaxJetIsolationA_; }
    int    isoTauEtaMin() const { return isoTauEtaMin_; }
    int    isoTauEtaMax() const { return isoTauEtaMax_; }
    std::string tauPUSType() const { return tauPUS_.type_; }
    const  std::vector<double> & tauPUSParams() const { return tauPUS_.dparams_; }
	double tauPUSParam(int ipar) const { return tauPUS_.dparams_.at(ipar); }

    l1t::LUT* tauIsolationLUT() { return tauIsolation_.LUT_.get(); }

    std::string tauCalibrationType() const { return tauCalibration_.type_; }
    std::vector<double> tauCalibrationParams() { return tauCalibration_.dparams_; }
    l1t::LUT* tauCalibrationLUT() { return tauCalibration_.LUT_.get(); }

    unsigned tauIsoAreaNrTowersEta()const{return tauIsoAreaNrTowersEta_;}
    unsigned tauIsoAreaNrTowersPhi()const{return tauIsoAreaNrTowersPhi_;}
    unsigned tauIsoVetoNrTowersPhi()const{return tauIsoVetoNrTowersPhi_;}


    void setTauLsb(double lsb) { tauLsb_ = lsb; }
    void setTauSeedThreshold(double thresh) { tauSeedThreshold_ = thresh; }
    void setTauNeighbourThreshold(double thresh) { tauNeighbourThreshold_ = thresh; }
    void setTauMaxPtTauVeto(double limit) { tauMaxPtTauVeto_ = limit; }
    void setTauMinPtJetIsolationB(double limit) { tauMinPtJetIsolationB_ = limit; }
    void setTauMaxJetIsolationB(double limit) { tauMaxJetIsolationB_ = limit; }
    void setTauMaxJetIsolationA(double cutValue) { tauMaxJetIsolationA_ = cutValue; }
    void setIsoTauEtaMin(int value) { isoTauEtaMin_ = value; }
    void setIsoTauEtaMax(int value) { isoTauEtaMax_ = value; }
    void setTauPUSType(std::string type) { tauPUS_.type_ = type; }
    void setTauIsolationLUT(std::shared_ptr<LUT> lut) { tauIsolation_.LUT_ = lut; }

    void setTauCalibrationType(std::string type) { tauCalibration_.type_ = type; }
    void setTauIsoAreaNrTowersEta(unsigned iTauIsoAreaNrTowersEta){tauIsoAreaNrTowersEta_=iTauIsoAreaNrTowersEta;}
    void setTauIsoAreaNrTowersPhi(unsigned iTauIsoAreaNrTowersPhi){tauIsoAreaNrTowersPhi_=iTauIsoAreaNrTowersPhi;}
    void setTauIsoVetoNrTowersPhi(unsigned iTauIsoVetoNrTowersPhi){tauIsoVetoNrTowersPhi_=iTauIsoVetoNrTowersPhi;}

    void setTauCalibrationParams(std::vector<double> params) { tauCalibration_.dparams_ = params; }
    void setTauCalibrationLUT(std::shared_ptr<LUT> lut) { tauCalibration_.LUT_ = lut; }
    
	void setTauPUSParams(const std::vector<double> & params) { tauPUS_.dparams_ = params; }

    // jets
    double jetLsb() const { return jetLsb_; }
    double jetSeedThreshold() const { return jetSeedThreshold_; }
    double jetNeighbourThreshold() const { return jetNeighbourThreshold_; }
    std::string jetPUSType() const { return jetPUS_.type_; }
    std::vector<double> jetPUSParams() { return jetPUS_.dparams_; }
    std::string jetCalibrationType() const { return jetCalibration_.type_; }
    std::vector<double> jetCalibrationParams() { return jetCalibration_.dparams_; }

    void setJetLsb(double lsb) { jetLsb_ = lsb; }
    void setJetSeedThreshold(double thresh) { jetSeedThreshold_ = thresh; }
    void setJetNeighbourThreshold(double thresh) { jetNeighbourThreshold_ = thresh; }
    void setJetPUSType(std::string type) { jetPUS_.type_ = type; }
    void setJetPUSParams(std::vector<double> params) { jetPUS_.dparams_ = params; }
    void setJetCalibrationType(std::string type) { jetCalibration_.type_ = type; }
    void setJetCalibrationParams(std::vector<double> params) { jetCalibration_.dparams_ = params; }


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
    l1t::LUT* centralityLUT() { return centrality_.LUT_.get(); }
    void setCentralityLUT(std::shared_ptr<LUT> lut) { centrality_.LUT_ = lut; }

    // HI Q2
    l1t::LUT* q2LUT() { return q2_.LUT_.get(); }
    void setQ2LUT(std::shared_ptr<LUT> lut) { q2_.LUT_ = lut; }

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }


  private:

    // AlgParams:  generic container for algorithm parameters,
    // including type name, LUTs, and vectors of various types.
    struct AlgParams {    
      std::string type_;
      unsigned version_;
      std::shared_ptr<l1t::LUT> LUT_;
      std::vector<double> dparams_;
      std::vector<unsigned> uparams_;
      std::vector<int> iparams_;
      std::vector<std::string> sparams_;
      AlgParams(){ type_="unspecified"; version_=0; }
    };

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

    // Unspecified Tower Algorithms:  
    std::vector<AlgParams> towerAlgs_; 

    /* Regions */

    // Region LSB
    double regionLsb_;

    // PUS scheme, parameters, and LUT
    AlgParams regionPUS_;

    // Unspecified Region Alogithms:  type, params, LUT
    std::vector<AlgParams> regionAlgs_;

    /* EG */

    // EG LSB
    double egLsb_;

    // Et threshold on EG seed tower
    double egSeedThreshold_;

    // Et threshold on EG neighbour tower(s)
    double egNeighbourThreshold_;

    // Et threshold on HCAL for H/E computation
    double egHcalThreshold_;

    // EG Trimmed shapes (indexed by |ieta|, shape)
    AlgParams egTrimming_;

    // EG maximum value of HCAL Et
    double egMaxHcalEt_;

    // Et threshold to remove the H/E cut from the EGammas
    double egMaxPtHOverE_;

    // EG maximum values of H/E (indexed by |ieta|, ??)
    AlgParams egMaxHOverE_;

    // Compress shapes
    AlgParams egCompressShapes_;

    // Shape identification bits (indexed by |ieta|, shape)
    AlgParams egShapeId_;

    // Range of jet isolation for EG (in rank!) (Stage1Layer2)
    int egMinPtJetIsolation_;
    int egMaxPtJetIsolation_;

    // Range of 3x3 HoE isolation for EG (in rank!) (Stage1Layer2)
    int egMinPtHOverEIsolation_;
    int egMaxPtHOverEIsolation_;

    // isolation area in eta is seed tower +/- <=egIsoAreaNrTowersPhi
    unsigned egIsoAreaNrTowersEta_;

    // isolation area in phi is seed tower +/- <=egIsoAreaNrTowersPhi
    unsigned egIsoAreaNrTowersPhi_;

    // veto region is seed tower +/- <=egIsoVetoNrTowersPhi
    unsigned egIsoVetoNrTowersPhi_;

    // EG calibration: type, parameters, and LUT
    AlgParams egCalibration_;

    // EG isolation PUS:  type, params, and LUT
    AlgParams egPUS_;

    // EG isolation:  type, params, and LUT (indexed by eta, Et ?)
    AlgParams egIsolation_;

    // Unspecified EG Algorithms:  type, params, LUT
    std::vector<AlgParams> egAlgs_;

    /* Tau */

    // Tau LSB
    double tauLsb_;

    // Et threshold on tau seed tower
    double tauSeedThreshold_;

    // Et threshold on tau neighbour towers
    double tauNeighbourThreshold_;

    // Et limit when to switch off tau veto requirement
    double tauMaxPtTauVeto_;

    // Et limit when to switch off tau isolation requirement
    double tauMinPtJetIsolationB_;

    // Et jet isolation limit for Taus (Stage1Layer2)
    double tauMaxJetIsolationB_;

    // Relative jet isolation cut for Taus (Stage1Layer2)
    double tauMaxJetIsolationA_;

    // Eta min and max for Iso-Tau collections (Stage1Layer2)
    int isoTauEtaMin_;
    int isoTauEtaMax_;
    
    // isolation area in eta is seed tower +/- <=tauIsoAreaNrTowersEta
    unsigned tauIsoAreaNrTowersEta_;

    // isolation area in phi is seed tower +/- <=tauIsoAreaNrTowersPhi
    unsigned tauIsoAreaNrTowersPhi_;

    // veto region is seed tower +/- <=tauIsoVetoNrTowersPhi
    unsigned tauIsoVetoNrTowersPhi_;
    
    // Tau PUS (currently applied to isolated taus only)
    AlgParams tauPUS_;

    // Tau isolation LUT (indexed by eta, Et ?)
    AlgParams tauIsolation_;

    // Tau calibration:  type, params, and LUT
    AlgParams tauCalibration_;

    // Unspecified Tau Algorithms:
    std::vector<AlgParams> tauAlgs_;

    /* Jets */

    // Jet LSB
    double jetLsb_;

    // Et threshold on jet seed tower/region
    double jetSeedThreshold_;

    // Et threshold on neighbouring towers/regions
    double jetNeighbourThreshold_;

    // jet PUS type, params, and LUT 
    // (type "None" means no PU)
    AlgParams jetPUS_;

    // jet calibration scheme 
    // (type "None" means no JEC)
    AlgParams jetCalibration_;

    // Unspecified Jet Algorithms:
    std::vector<AlgParams> jetAlgs_;

    /* Sums */

    // EtSum LSB
    double etSumLsb_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMin_;

    // maximum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMax_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<double> etSumEtThreshold_;

    // Unspecified ET Sum Algorithms:
    std::vector<AlgParams> etSumAlgs_;

    /* HI */

    // centrality LUT
    AlgParams centrality_;

    // Q2 LUT
    AlgParams q2_;

    // Unspecified HI algorithms:
    std::vector<AlgParams> hiAlgs_;

    /* Attic:  Reserved for future use */

    // Attic variables of last resort:
    unsigned atticVersion;  // UNUSED:  Reserved for future use 
    std::vector<AlgParams> atticAlgs_;
  };

}// namespace
#endif
