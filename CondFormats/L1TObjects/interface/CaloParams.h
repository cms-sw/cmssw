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

#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

namespace l1t {


  class CaloParams {

  public:

    enum { Version = 1 };

    class Node {
    public:
      std::string type_;
      unsigned version_;
      l1t::LUT LUT_;
      std::vector<double> dparams_;
      std::vector<unsigned> uparams_;
      std::vector<int> iparams_;
      std::vector<std::string> sparams_;
      Node(){ type_="unspecified"; version_=0; }
      COND_SERIALIZABLE;
    };

    class TowerParams{
    public:
      /* Towers */

      // LSB of HCAL scale
      double lsbH_;

      // LSB of ECAL scale
      double lsbE_;

      // LSB of ECAL+HCAL sum scale
      double lsbSum_;

      // number of bits for HCAL encoding
      int nBitsH_;

      // number of bits for ECAL encoding
      int nBitsE_;

      // number of bits for ECAL+HCAL sum encoding
      int nBitsSum_;

      // number of bits for ECAL/HCAL ratio encoding
      int nBitsRatio_;

      // bitmask for storing HCAL Et in  object
      int maskH_;

      // bitmask for storing ECAL ET in  object
      int maskE_;

      // bitmask for storing ECAL+HCAL sum in  object
      int maskSum_;

      // bitmask for storing ECAL/HCAL ratio in  object
      int maskRatio_;

      // turn encoding on/off
      bool doEncoding_;

      TowerParams() : lsbH_(0), lsbE_(0), lsbSum_(0),
		      nBitsH_(0), nBitsE_(0), nBitsSum_(0), nBitsRatio_(0),
		      maskH_(0), maskE_(0), maskSum_(0), maskRatio_(0), 
		      doEncoding_(false)
      { /* no-op */}

      COND_SERIALIZABLE;
    };

    class EgParams {
    public:
      // EG LSB
      double lsb_;

      // Et threshold on EG seed tower
      double seedThreshold_;

      // Et threshold on EG neighbour tower(s)
      double neighbourThreshold_;

      // Et threshold on HCAL for H/E computation
      double hcalThreshold_;

      // EG maximum value of HCAL Et
      double maxHcalEt_;

      // Et threshold to remove the H/E cut from the EGammas
      double maxPtHOverE_;

      // Range of jet isolation for EG (in rank!) (Stage1Layer2)
      int minPtJetIsolation_;
      int maxPtJetIsolation_;

      // Range of 3x3 HoE isolation for EG (in rank!) (Stage1Layer2)
      int minPtHOverEIsolation_;
      int maxPtHOverEIsolation_;

      // isolation area in eta is seed tower +/- <=egIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersEta_;

      // isolation area in phi is seed tower +/- <=egIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersPhi_;

      // veto region is seed tower +/- <=egIsoVetoNrTowersPhi
      unsigned isoVetoNrTowersPhi_;

      EgParams() : lsb_(0), seedThreshold_(0), neighbourThreshold_(0), hcalThreshold_(0), maxHcalEt_(0), maxPtHOverE_(0), 
		   minPtJetIsolation_(0), maxPtJetIsolation_(0), minPtHOverEIsolation_(0), maxPtHOverEIsolation_(0), 
		   isoAreaNrTowersEta_(0), isoAreaNrTowersPhi_(0), isoVetoNrTowersPhi_(0)
      { /* no-op */ }

      COND_SERIALIZABLE;
    };


    class TauParams {
    public:
      // Tau LSB
      double lsb_;

      // Et threshold on tau seed tower
      double seedThreshold_;

      // Et threshold on tau neighbour towers
      double neighbourThreshold_;

      // Et limit when to switch off tau veto requirement
      double maxPtTauVeto_;

      // Et limit when to switch off tau isolation requirement
      double minPtJetIsolationB_;

      // Et jet isolation limit for Taus (Stage1Layer2)
      double maxJetIsolationB_;

      // Relative jet isolation cut for Taus (Stage1Layer2)
      double maxJetIsolationA_;

      // Eta min and max for Iso-Tau collections (Stage1Layer2)
      int isoEtaMin_;
      int isoEtaMax_;

      // isolation area in eta is seed tower +/- <=tauIsoAreaNrTowersEta
      unsigned isoAreaNrTowersEta_;

      // isolation area in phi is seed tower +/- <=tauIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersPhi_;

      // veto region is seed tower +/- <=tauIsoVetoNrTowersPhi
      unsigned isoVetoNrTowersPhi_;

      TauParams() : lsb_(0), seedThreshold_(0), neighbourThreshold_(0), maxPtTauVeto_(0), 
		    minPtJetIsolationB_(0), maxJetIsolationB_(0), maxJetIsolationA_(0),
		    isoEtaMin_(0), isoEtaMax_(0), 
		    isoAreaNrTowersEta_(0), isoAreaNrTowersPhi_(0), isoVetoNrTowersPhi_(0)
      { /* no-op */ }

      COND_SERIALIZABLE;
    };

    class JetParams {
    public:
      // Jet LSB
      double lsb_;

      // Et threshold on jet seed tower/region
      double seedThreshold_;

      // Et threshold on neighbouring towers/regions
      double neighbourThreshold_;

      JetParams() : lsb_(0), seedThreshold_(0), neighbourThreshold_(0) { /* no-op */ }

      COND_SERIALIZABLE;
    };


    // DO NOT ADD ENTRIES ANYWHERE BUT DIRECTLY BEFORE "NUM_CALOPARAMNODES"
    enum { regionPUS=0,
	   egTrimming=1, egMaxHOverE=2, egCompressShapes=3, egShapeId=4, egCalibration=5, egPUS=6, egIsolation=7,
	   tauCalibration=8, tauPUS=9, tauIsolation=10,
	   jetPUS=11, jetCalibration=12,
	   hiCentrality=13, hiQ2=14, 
	   tauEtToHFRingEt=15,
	   NUM_CALOPARAMNODES=16
    };

    CaloParams() { version_=Version; pnode_.resize(NUM_CALOPARAMNODES); }
    ~CaloParams() {}

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

    void setRegionLsb(double lsb) { regionLsb_ = lsb; }
    void setRegionPUSType(std::string type) { pnode_[regionPUS].type_ = type; }
    void setRegionPUSParams(const std::vector<double> & params) { pnode_[regionPUS].dparams_ = params; }

    // EG
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



    l1t::LUT* egIsolationLUT() { return &pnode_[egIsolation].LUT_; }
    std::string egCalibrationType() const { return pnode_[egCalibration].type_; }
    std::vector<double> egCalibrationParams() { return pnode_[egCalibration].dparams_; }
    l1t::LUT* egCalibrationLUT() { return &pnode_[egCalibration].LUT_; }

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
    void setEgIsolationLUT(const l1t::LUT & lut) { pnode_[egIsolation].LUT_ = lut; }
    void setEgCalibrationType(std::string type) { pnode_[egCalibration].type_ = type; }
    void setEgCalibrationParams(std::vector<double> params) { pnode_[egCalibration].dparams_ = params; }
    void setEgCalibrationLUT(const l1t::LUT & lut) { pnode_[egCalibration].LUT_ = lut; }

    // tau
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

    l1t::LUT* tauEtToHFRingEtLUT() { return &pnode_[tauEtToHFRingEt].LUT_; }

    unsigned tauIsoAreaNrTowersEta()const{return taup_.isoAreaNrTowersEta_;}
    unsigned tauIsoAreaNrTowersPhi()const{return taup_.isoAreaNrTowersPhi_;}
    unsigned tauIsoVetoNrTowersPhi()const{return taup_.isoVetoNrTowersPhi_;}


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
    void setTauPUSParams(const std::vector<double> & params) { pnode_[tauPUS].dparams_ = params; }

    void setTauEtToHFRingEtLUT(const l1t::LUT & lut) { pnode_[tauEtToHFRingEt].LUT_ = lut; }

    // jets
    double jetLsb() const { return jetp_.lsb_; }
    double jetSeedThreshold() const { return jetp_.seedThreshold_; }
    double jetNeighbourThreshold() const { return jetp_.neighbourThreshold_; }
    std::string jetPUSType() const { return pnode_[jetPUS].type_; }
    std::vector<double> jetPUSParams() { return pnode_[jetPUS].dparams_; }
    std::string jetCalibrationType() const { return pnode_[jetCalibration].type_; }
    std::vector<double> jetCalibrationParams() { return pnode_[jetCalibration].dparams_; }
    l1t::LUT* jetCalibrationLUT() { return &pnode_[jetCalibration].LUT_; }

    void setJetLsb(double lsb) { jetp_.lsb_ = lsb; }
    void setJetSeedThreshold(double thresh) { jetp_.seedThreshold_ = thresh; }
    void setJetNeighbourThreshold(double thresh) { jetp_.neighbourThreshold_ = thresh; }
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
    l1t::LUT * centralityLUT() { return &pnode_[hiCentrality].LUT_; }
    void setCentralityLUT(const l1t::LUT & lut) { pnode_[hiCentrality].LUT_ = lut; }

    // HI Q2
    l1t::LUT * q2LUT() { return &pnode_[hiQ2].LUT_; }
    void setQ2LUT(const l1t::LUT & lut) { pnode_[hiQ2].LUT_ = lut; }

    // print parameters to stream:
    void print(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream& o, const CaloParams & p) { p.print(o); return o; }

  private:
    unsigned version_;

    std::vector<Node> pnode_;

    TowerParams towerp_;

    // Region LSB
    double regionLsb_;

    EgParams egp_;
    TauParams taup_;
    JetParams jetp_;

    /* Sums */

    // EtSum LSB
    double etSumLsb_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMin_;

    // maximum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMax_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<double> etSumEtThreshold_;


    COND_SERIALIZABLE;
  };

}// namespace
#endif
