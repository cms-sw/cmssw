#include <vector>
#include "CondFormats/L1TObjects/interface/CaloParams.h"

class CaloParams_PUBLIC : public l1t::CaloParams {
public:
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

    CaloParams_PUBLIC(const l1t::CaloParams &p) : l1t::CaloParams(p){
        version_ = l1t::CaloParams::version_;
        pnode_   = l1t::CaloParams::pnode_;
        towerp_  = l1t::CaloParams::towerp_;
        regionLsb_ = l1t::CaloParams::regionLsb_;
        egp_     = l1t::CaloParams::egp_;
        taup_    = l1t::CaloParams::taup_;
        jetp_    = l1t::CaloParams::jetp_;
        etSumLsb_ = l1t::CaloParams::etSumLsb_;
        etSumEtaMin_ = l1t::CaloParams::etSumEtaMin_;
        etSumEtaMax_ = l1t::CaloParams::etSumEtaMax_;
        etSumEtThreshold_ = l1t::CaloParams::etSumEtThreshold_;
    }
};

class LUT_PUBLIC {
public:
    unsigned int nrBitsAddress_; //technically redundant with addressMask
    unsigned int nrBitsData_;//technically redundant with dataMask
    unsigned int addressMask_;
    unsigned int dataMask_;
   
    std::vector<int> data_;
};

bool operator == (const std::vector<int>& a1, const std::vector<int>& a2){
if( a1.size() != a2.size() ) return false;
for(unsigned i=0; i<a1.size(); i++)
   if( a1[i] != a2[i] ) return false;
return true;
}

bool operator == (const l1t::CaloParams::TowerParams &a1, const l1t::CaloParams::TowerParams &a2){
    return (
      a1.lsbH_ == a2.lsbH_ && 
      a1.lsbE_ == a2.lsbE_ &&
      a1.lsbSum_ == a2.lsbSum_ &&
      a1.nBitsH_ == a2.nBitsH_ &&
      a1.nBitsE_ == a2.nBitsE_ &&
      a1.nBitsSum_ == a2.nBitsSum_ &&
      a1.nBitsRatio_ == a2.nBitsRatio_ &&
      a1.maskH_ == a2.maskH_ &&
      a1.maskE_ == a2.maskE_ &&
      a1.maskSum_ == a2.maskSum_ &&
      a1.maskRatio_ == a2.maskRatio_ &&
      a1.doEncoding_ == a2.doEncoding_ );
}

bool operator == (const l1t::LUT& b1, const l1t::LUT& b2){
    const LUT_PUBLIC &a1 = reinterpret_cast<const LUT_PUBLIC&>(b1);
    const LUT_PUBLIC &a2 = reinterpret_cast<const LUT_PUBLIC&>(b2);

    bool a = ( a1.nrBitsAddress_ == a2.nrBitsAddress_ );
    bool d = ( a1.nrBitsData_ == a2.nrBitsData_ );
    bool m = ( a1.addressMask_ == a2.addressMask_ ) ;
    bool mm = ( a1.dataMask_ == a2.dataMask_ ) ;
    bool v = (a1.data_ == a2.data_);
    std::cout<<"a="<<a<<" d="<<d<<" m="<<m<<" mm="<<mm<<" v="<<v<<std::endl;
    return a && d && m && mm && v;
}

bool operator == (const l1t::CaloParams::Node& a1, const l1t::CaloParams::Node& a2){
    bool t =  ( a1.type_ == a2.type_ );
    bool v =  ( a1.version_ == a2.version_ );
    bool l =  ( a1.LUT_ == a2.LUT_ );
    bool d =  ( a1.dparams_ == a2.dparams_) ;
    bool u =  ( a1.uparams_ == a2.uparams_) ;
    bool i =  ( a1.iparams_ == a2.iparams_) ;
    bool s =  ( a1.sparams_ == a2.sparams_) ;
    std::cout<<"t="<<t<<" v="<<v<<" l="<<l<<" d="<<d<<" u="<<u<<" i="<<i<<" s="<<s<<std::endl;
    return t && v && l && d && u && i && s;
}

bool operator == (const std::vector<l1t::CaloParams::Node>& a1, const std::vector<l1t::CaloParams::Node>& a2){
if( a1.size() != a2.size() ) return false;
for(unsigned i=0; i<a1.size(); i++){
   
   if( !(a1[i] == a2[i]) ) return false;
}
return true;
}

bool operator == (const l1t::CaloParams& b1, const l1t::CaloParams& b2){

    CaloParams_PUBLIC a1(b1);
    CaloParams_PUBLIC a2(b2);

    bool v = (a1.version_ == a2.version_);
    std::cout<<" version_: " << v << std::endl;

    bool pn = (a1.pnode_[1] == a2.pnode_[1]);
    std::cout<<" pnode_: " << pn << std::endl;

    bool tp = (a1.towerp_ == a2.towerp_);
    std::cout<<" towerp_: " << tp << std::endl;
    bool rlsb = (a1.regionLsb_ == a2.regionLsb_);
    std::cout<<" regionLsb_: "<< rlsb << std::endl;

    bool egp =             ((a1.egp_.lsb_ == a2.egp_.lsb_) &&
                            (a1.egp_.seedThreshold_ == a2.egp_.seedThreshold_) &&
                            (a1.egp_.neighbourThreshold_== a2.egp_.neighbourThreshold_) &&
                            (a1.egp_.hcalThreshold_ == a2.egp_.hcalThreshold_) &&
                            (a1.egp_.maxHcalEt_== a2.egp_.maxHcalEt_) &&
                            (a1.egp_.maxPtHOverE_== a2.egp_.maxPtHOverE_) &&
                            (a1.egp_.minPtJetIsolation_== a2.egp_.minPtJetIsolation_) &&
                            (a1.egp_.maxPtJetIsolation_== a2.egp_.maxPtJetIsolation_) &&
                            (a1.egp_.isoAreaNrTowersEta_== a2.egp_.isoAreaNrTowersEta_) &&
                            (a1.egp_.isoAreaNrTowersPhi_== a2.egp_.isoAreaNrTowersPhi_) &&
                            (a1.egp_.isoVetoNrTowersPhi_== a2.egp_.isoVetoNrTowersPhi_) );
    std::cout<<" egp_: "<< egp << std::endl;

    bool taup =             ((a1.taup_.lsb_ == a2.taup_.lsb_) &&
                            (a1.taup_.seedThreshold_ == a2.taup_.seedThreshold_) &&
                            (a1.taup_.neighbourThreshold_ == a2.taup_.neighbourThreshold_) &&
                            (a1.taup_.maxPtTauVeto_ == a2.taup_.maxPtTauVeto_) &&
                            (a1.taup_.minPtJetIsolationB_ == a2.taup_.minPtJetIsolationB_) &&
                            (a1.taup_.maxJetIsolationB_ == a2.taup_.maxJetIsolationB_) &&
                            (a1.taup_.maxJetIsolationA_ == a2.taup_.maxJetIsolationA_) &&
                            (a1.taup_.isoEtaMin_ == a2.taup_.isoEtaMin_) &&
                            (a1.taup_.isoEtaMax_ == a2.taup_.isoEtaMax_) &&
                            (a1.taup_.isoAreaNrTowersEta_== a2.taup_.isoAreaNrTowersEta_) &&
                            (a1.taup_.isoAreaNrTowersPhi_== a2.taup_.isoAreaNrTowersPhi_) &&
                            (a1.taup_.isoVetoNrTowersPhi_== a2.taup_.isoVetoNrTowersPhi_));
    std::cout<<" taup_: "<< taup << std::endl;

    bool jetp =            ((a1.jetp_.lsb_ == a2.jetp_.lsb_) &&
                            (a1.jetp_.seedThreshold_ == a2.jetp_.seedThreshold_) &&
                            (a1.jetp_.neighbourThreshold_ == a2.jetp_.neighbourThreshold_));
    std::cout<<" jetp_: " << jetp << std::endl;

    bool etslsb = (a1.etSumLsb_ == a2.etSumLsb_);
    std::cout<<" etSumLsb_: "<< etslsb << std::endl;
    bool etsemn = (a1.etSumEtaMin_ == a2.etSumEtaMin_);
    std::cout<<" etSumEtaMin_: "<< etsemn << std::endl;

    bool etsemx = (a1.etSumEtaMax_ == a2.etSumEtaMax_);
    std::cout<<" etSumEtaMax_: "<< etsemx << std::endl;
    bool sett = (a1.etSumEtThreshold_ == a2.etSumEtThreshold_);
    std::cout<<" etSumEtThreshold_: "<< sett << std::endl;

    return v && pn && tp && rlsb && egp && taup && jetp && etslsb && etsemn && etsemx && sett;
}
