#ifndef DataFormats_METReco_PUSubMETData_h
#define DataFormats_METReco_PUSubMETData_h

/** \class MVAMEtData.h
 *
 * Storage of PFCandidate and PFJet id. information used in MVA MET calculation.
 *
 * \authors Phil Harris, CERN
 *          Christian Veelken, LLR
 *
 *
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/SigInputObj.h"

namespace reco 
{

  class PUSubMETCandInfo {

    //functions
  public:
    
    PUSubMETCandInfo();
    ~PUSubMETCandInfo();
    
    bool operator<(const reco::PUSubMETCandInfo&) const;


    //Access functions ================
    const reco::Candidate::LorentzVector& p4() const {return p4_;};
    const float dZ() const {return dZ_;};
    
    int type() const {return type_;};
    int charge() const {return charge_;};

    //Jet specific
    bool isWithinJet() const {return isWithinJet_;};
    float passesLooseJetId() const {return passesLooseJetId_;};
    float offsetEnCorr() const {return offsetEnCorr_;};
    float mva() const {return mva_;};
    float chargedEnFrac() const {return chargedEnFrac_;};

    const metsig::SigInputObj& metSignObj() const {return pfMEtSignObj_;};

    //setting functions ================
    void setP4( const reco::Candidate::LorentzVector p4 ) {p4_ = p4;};
    void setDZ(float dZ) {dZ_ = dZ;};

    void setType(int type) {type_ = type;};
    void setCharge(int charge) {charge_ = charge;};

    //Jet specific
    void setIsWithinJet(bool isWJ) {isWithinJet_ = isWJ;};
    void setPassesLooseJetId(float jetId) {passesLooseJetId_ = jetId;};
    void setOffsetEnCorr(float offset) {offsetEnCorr_ = offset;};
    void setMvaVal(float mva) {mva_ = mva;};
    void setChargedEnFrac(float chEnF) {chargedEnFrac_ = chEnF;};

    void setMEtSignObj(metsig::SigInputObj msig) {pfMEtSignObj_ = msig;};

    //variables =================
  public:

    enum { kUndefined=0, kHS, kChHS, kPU, kChPU, kNeutral };

  private:

    reco::Candidate::LorentzVector p4_;
    float dZ_;
    
    int type_;
    int charge_;

    //Jet specific
    bool isWithinJet_;
    float passesLooseJetId_;
    float offsetEnCorr_;
    float mva_;
    float chargedEnFrac_;
    
    metsig::SigInputObj pfMEtSignObj_; // contribution of this PFJet to PFMET significance matrix

    
  };

}

#endif
