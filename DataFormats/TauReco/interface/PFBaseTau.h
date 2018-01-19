#ifndef DataFormats_TauReco_PFBaseTau_h
#define DataFormats_TauReco_PFBaseTau_h

/* class PFBaseTau
 * Based on regular PFTau class, but uses CandidatePtrs instead of 
 * CandidatePtrs and base refs to reco::Jets instead of refs to reco::Jets
 * to be compatible with miniAOD inputs.
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFBaseTauFwd.h"
// #include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/RecoTauPiZeroFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"

#include <iostream>
#include <limits>

namespace reco { namespace tau {
  template<class T, typename U, typename V>
  class RecoTauConstructor;
  template<class T>
  class PFRecoTauGenericEnergyAlgorithmPlugin;
}} 

namespace reco {
  class PFBaseTauDiscriminator;
}

namespace pat {
    class Jet;
}

namespace reco {

class PFBaseTau : public BaseTau {
  public:
    enum hadronicDecayMode {
      kNull = -1,
      kOneProng0PiZero,
      kOneProng1PiZero,
      kOneProng2PiZero,
      kOneProng3PiZero,
      kOneProngNPiZero,
      kTwoProng0PiZero,
      kTwoProng1PiZero,
      kTwoProng2PiZero,
      kTwoProng3PiZero,
      kTwoProngNPiZero,
      kThreeProng0PiZero,
      kThreeProng1PiZero,
      kThreeProng2PiZero,
      kThreeProng3PiZero,
      kThreeProngNPiZero,
      kRareDecayMode
    };

    typedef std::vector<pat::Jet> TauJetCollection;
    typedef PFBaseTauDiscriminator TauDiscriminator;
    typedef Candidate TauCandidateType;

    PFBaseTau();
    PFBaseTau(Charge q,const LorentzVector &,const Point & = Point( 0, 0, 0 ) );
    ~PFBaseTau() override {};
    PFBaseTau* clone() const override;

    const JetBaseRef& jetRef() const;
    void setjetRef(const JetBaseRef&);

    // // functions to access the PFTauTagInfoRef used by HLT
    // const PFTauTagInfoRef& pfTauTagInfoRef() const;
    // void setpfTauTagInfoRef(const PFTauTagInfoRef);

    PFRecoTauChargedHadronRef leadTauChargedHadronCandidate() const;
    const CandidatePtr& leadPFChargedHadrCand() const;
    const CandidatePtr& leadPFNeutralCand() const;
    //Can be either the charged or the neutral one
    const CandidatePtr& leadPFCand() const;

    void setleadPFChargedHadrCand(const CandidatePtr&);
    void setleadPFNeutralCand(const CandidatePtr&);
    void setleadPFCand(const CandidatePtr&);

    /// Signed transverse impact parameter significance of the Track
    /// associated to the leading charged Candidate
    float leadPFChargedHadrCandsignedSipt() const;
    void setleadPFChargedHadrCandsignedSipt(const float&);

    /// Candidates in signal region
    const std::vector<reco::CandidatePtr>& signalPFCands() const;
    void setsignalPFCands(const std::vector<reco::CandidatePtr>&);

    /// Charged hadrons in signal region
    const std::vector<reco::CandidatePtr>& signalPFChargedHadrCands() const;
    void setsignalPFChargedHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Neutral hadrons in signal region
    const std::vector<reco::CandidatePtr>& signalPFNeutrHadrCands() const;
    void setsignalPFNeutrHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Gamma candidates in signal region
    const std::vector<reco::CandidatePtr>& signalPFGammaCands() const;
    void setsignalPFGammaCands(const std::vector<reco::CandidatePtr>&);

    /// Candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationPFCands() const;
    void setisolationPFCands(const std::vector<reco::CandidatePtr>&);

    /// Charged candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationPFChargedHadrCands() const;
    void setisolationPFChargedHadrCands(const std::vector<reco::CandidatePtr>&);

    //// Neutral hadrons in isolation region
    const std::vector<reco::CandidatePtr>& isolationPFNeutrHadrCands() const;
    void setisolationPFNeutrHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Gamma candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationPFGammaCands() const;
    void setisolationPFGammaCands(const std::vector<reco::CandidatePtr>&);

    /// Sum of charged hadron candidate PT in isolation cone; returns NaN
    /// if isolation region is undefined.
    float isolationPFChargedHadrCandsPtSum() const;
    void setisolationPFChargedHadrCandsPtSum(const float&);

    /// Sum of gamma candidate PT in isolation cone; returns NaN
    /// if isolation region is undefined.
    float isolationPFGammaCandsEtSum() const;
    void setisolationPFGammaCandsEtSum(const float&);

    /// Et of the highest Et HCAL PFCluster
    float maximumHCALPFClusterEt() const;
    void setmaximumHCALPFClusterEt(const float&);

    /// Retrieve the association of signal region gamma candidates into candidate PiZeros
    const std::vector<RecoTauPiZero>& signalPiZeroCandidates() const;
    void setsignalPiZeroCandidates(std::vector<RecoTauPiZero>);
    void setSignalPiZeroCandidatesRefs(RecoTauPiZeroRefVector);

    /// Retrieve the association of isolation region gamma candidates into candidate PiZeros
    const std::vector<RecoTauPiZero>& isolationPiZeroCandidates() const;
    void setisolationPiZeroCandidates(std::vector<RecoTauPiZero>);
    void setIsolationPiZeroCandidatesRefs(RecoTauPiZeroRefVector);

    /// Retrieve the association of signal region PF candidates into candidate PFRecoTauChargedHadrons
    const std::vector<PFRecoTauChargedHadron>& signalTauChargedHadronCandidates() const;
    void setSignalTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron>);
    void setSignalTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector);

    /// Retrieve the association of isolation region PF candidates into candidate PFRecoTauChargedHadron
    const std::vector<PFRecoTauChargedHadron>& isolationTauChargedHadronCandidates() const;
    void setIsolationTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron>);
    void setIsolationTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector);

    /// Retrieve the identified hadronic decay mode according to the number of
    /// charged and piZero candidates in the signal cone
    hadronicDecayMode decayMode() const;
    void setDecayMode(const hadronicDecayMode&);

    /// Effect of eta and phi correction of strip on mass of tau candidate
    float bendCorrMass() const { return bendCorrMass_; }
    void setBendCorrMass(float bendCorrMass) { bendCorrMass_ = bendCorrMass; }

    /// Size of signal cone
    double signalConeSize() const { return signalConeSize_; }
    void setSignalConeSize(double signalConeSize) { signalConeSize_ = signalConeSize; }  

    //Electron rejection
    float emFraction() const; // Ecal/Hcal Cluster Energy
    float hcalTotOverPLead() const; // total Hcal Cluster E / leadPFChargedHadron P
    float hcalMaxOverPLead() const; // max. Hcal Cluster E / leadPFChargedHadron P
    float hcal3x3OverPLead() const; // Hcal Cluster E in R<0.184 around Ecal impact point of leading track / leadPFChargedHadron P
    float ecalStripSumEOverPLead() const; // Simple BremsRecovery Sum E / leadPFChargedHadron P
    float bremsRecoveryEOverPLead() const; // BremsRecovery Sum E / leadPFChargedHadron P
    reco::TrackRef electronPreIDTrack() const; // Ref to KF track from Electron PreID
    float electronPreIDOutput() const; // BDT output from Electron PreID
    bool electronPreIDDecision() const; // Decision from Electron PreID

    void setemFraction(const float&);
    void sethcalTotOverPLead(const float&);
    void sethcalMaxOverPLead(const float&);
    void sethcal3x3OverPLead(const float&);
    void setecalStripSumEOverPLead(const float&);
    void setbremsRecoveryEOverPLead(const float&);
    void setelectronPreIDTrack(const reco::TrackRef&);
    void setelectronPreIDOutput(const float&);
    void setelectronPreIDDecision(const bool&);

    float caloComp() const;
    float segComp() const;
    bool muonDecision() const;
    void setCaloComp(const float&);
    void setSegComp(const float&);
    void setMuonDecision(const bool&);

    /// return the number of source Candidates
    /// ( the candidates used to construct this Candidate)
    /// in the case of taus, there is only one source candidate,
    /// which is the corresponding Jet
    size_type numberOfSourceCandidatePtrs() const override {return 1;}

    /// return a RefToBase to the source Candidates
    /// ( the candidates used to construct this Candidate)
    CandidatePtr sourceCandidatePtr( size_type i ) const override;

    /// prints information on this PFBaseTau
    void dump(std::ostream& out = std::cout) const;

  private:
    template<typename T, typename U, typename V>
    friend class tau::RecoTauConstructor;
    template<typename T>
    friend class tau::PFRecoTauGenericEnergyAlgorithmPlugin;

    //These are used by the friends
    std::vector<RecoTauPiZero>& signalPiZeroCandidatesRestricted();
    std::vector<RecoTauPiZero>& isolationPiZeroCandidatesRestricted();
    std::vector<PFRecoTauChargedHadron>& signalTauChargedHadronCandidatesRestricted();
    std::vector<PFRecoTauChargedHadron>& isolationTauChargedHadronCandidatesRestricted();

    // check overlap with another candidate
    bool overlap(const Candidate&) const override;

    bool muonDecision_;
    bool electronPreIDDecision_;

    // SIP
    float leadPFChargedHadrCandsignedSipt_;
    // Isolation variables
    float isolationPFChargedHadrCandsPtSum_;
    float isolationPFGammaCandsEtSum_;
    float maximumHCALPFClusterEt_;

    // Electron rejection variables
    float emFraction_;
    float hcalTotOverPLead_;
    float hcalMaxOverPLead_;
    float hcal3x3OverPLead_;
    float ecalStripSumEOverPLead_;
    float bremsRecoveryEOverPLead_;
    float electronPreIDOutput_;

    // Muon rejection variables
    float caloComp_;
    float segComp_;

    hadronicDecayMode decayMode_;

    float bendCorrMass_;

    float signalConeSize_; 

    reco::JetBaseRef jetRef_;
    // PFTauTagInfoRef PFTauTagInfoRef_;
    reco::CandidatePtr leadPFChargedHadrCand_;
    reco::CandidatePtr leadPFNeutralCand_;
    reco::CandidatePtr leadPFCand_;
    reco::TrackRef electronPreIDTrack_;

    // Signal candidates
    std::vector<reco::CandidatePtr> selectedSignalPFCands_;
    std::vector<reco::CandidatePtr> selectedSignalPFChargedHadrCands_;
    std::vector<reco::CandidatePtr> selectedSignalPFNeutrHadrCands_;
    std::vector<reco::CandidatePtr> selectedSignalPFGammaCands_;

    // Isolation candidates
    std::vector<reco::CandidatePtr> selectedIsolationPFCands_;
    std::vector<reco::CandidatePtr> selectedIsolationPFChargedHadrCands_;
    std::vector<reco::CandidatePtr> selectedIsolationPFNeutrHadrCands_;
    std::vector<reco::CandidatePtr> selectedIsolationPFGammaCands_;

    RecoTauPiZeroRefVector signalPiZeroCandidatesRefs_;
    RecoTauPiZeroRefVector isolationPiZeroCandidatesRefs_;

    PFRecoTauChargedHadronRefVector signalTauChargedHadronCandidatesRefs_;
    PFRecoTauChargedHadronRefVector isolationTauChargedHadronCandidatesRefs_;

    // Association of gamma candidates into PiZeros (transient)
    edm::AtomicPtrCache<std::vector<reco::RecoTauPiZero>> signalPiZeroCandidates_;
    edm::AtomicPtrCache<std::vector<reco::RecoTauPiZero>> isolationPiZeroCandidates_;

    // Association of PF candidates into PFRecoTauChargedHadrons (transient)
    edm::AtomicPtrCache<std::vector<reco::PFRecoTauChargedHadron>> signalTauChargedHadronCandidates_;
    edm::AtomicPtrCache<std::vector<reco::PFRecoTauChargedHadron>> isolationTauChargedHadronCandidates_;
};

std::ostream & operator<<(std::ostream& out, const PFBaseTau& c);

} // end namespace reco

#endif
