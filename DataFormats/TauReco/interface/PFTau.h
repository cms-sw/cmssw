#ifndef DataFormats_TauReco_PFTau_h
#define DataFormats_TauReco_PFTau_h

/* class PFTau
 * the object of this class is created by RecoTauTag/RecoTau PFRecoTauProducer EDProducer starting from the PFTauTagInfo object,
 *                          is a hadronic tau-jet candidate -built from a jet made employing a particle flow technique- that analysts manipulate;
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch), Evan Friis (evan.klose.friis@ucdavis.edu)
 * created: Jun 21 2007,
 * revised: Tue Aug 31 13:34:40 CEST 2010
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/TauReco/interface/RecoTauPiZeroFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"

#include <iostream>
#include <limits>

namespace reco {
  namespace tau {
    class RecoTauConstructor;
    class PFRecoTauEnergyAlgorithmPlugin;
  }  // namespace tau
}  // namespace reco

namespace reco {

  class PFTau : public BaseTau {
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

    PFTau();
    PFTau(Charge q, const LorentzVector&, const Point& = Point(0, 0, 0));
    ~PFTau() override{};
    PFTau* clone() const override;

    const JetBaseRef& jetRef() const;
    void setjetRef(const JetBaseRef&);

    // functions to access the PFTauTagInfoRef used by HLT
    const PFTauTagInfoRef& pfTauTagInfoRef() const;
    void setpfTauTagInfoRef(const PFTauTagInfoRef);

    PFRecoTauChargedHadronRef leadTauChargedHadronCandidate() const;
    const CandidatePtr& leadChargedHadrCand() const;
    const CandidatePtr& leadNeutralCand() const;
    //Can be either the charged or the neutral one
    const CandidatePtr& leadCand() const;

    void setleadChargedHadrCand(const CandidatePtr&);
    void setleadNeutralCand(const CandidatePtr&);
    void setleadCand(const CandidatePtr&);

    /// Signed transverse impact parameter significance of the Track
    /// associated to the leading charged PFCandidate
    float leadPFChargedHadrCandsignedSipt() const;
    void setleadPFChargedHadrCandsignedSipt(const float&);

    /// Candidates in signal region
    const std::vector<reco::CandidatePtr>& signalCands() const;
    void setsignalCands(const std::vector<reco::CandidatePtr>&);

    /// Charged hadrons in signal region
    const std::vector<reco::CandidatePtr>& signalChargedHadrCands() const;
    void setsignalChargedHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Neutral hadrons in signal region
    const std::vector<reco::CandidatePtr>& signalNeutrHadrCands() const;
    void setsignalNeutrHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Gamma candidates in signal region
    const std::vector<reco::CandidatePtr>& signalGammaCands() const;
    void setsignalGammaCands(const std::vector<reco::CandidatePtr>&);

    /// Candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationCands() const;
    void setisolationCands(const std::vector<reco::CandidatePtr>&);

    /// Charged candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationChargedHadrCands() const;
    void setisolationChargedHadrCands(const std::vector<reco::CandidatePtr>&);

    //// Neutral hadrons in isolation region
    const std::vector<reco::CandidatePtr>& isolationNeutrHadrCands() const;
    void setisolationNeutrHadrCands(const std::vector<reco::CandidatePtr>&);

    /// Gamma candidates in isolation region
    const std::vector<reco::CandidatePtr>& isolationGammaCands() const;
    void setisolationGammaCands(const std::vector<reco::CandidatePtr>&);

    /// Getters for different PFCandidates for PFTaus made from PFCandidates
    const PFCandidatePtr leadPFChargedHadrCand() const;
    const PFCandidatePtr leadPFNeutralCand() const;
    const PFCandidatePtr leadPFCand() const;
    const std::vector<reco::PFCandidatePtr>& signalPFCands() const;
    const std::vector<reco::PFCandidatePtr>& signalPFChargedHadrCands() const;
    const std::vector<reco::PFCandidatePtr>& signalPFNeutrHadrCands() const;
    const std::vector<reco::PFCandidatePtr>& signalPFGammaCands() const;
    const std::vector<reco::PFCandidatePtr>& isolationPFCands() const;
    const std::vector<reco::PFCandidatePtr>& isolationPFChargedHadrCands() const;
    const std::vector<reco::PFCandidatePtr>& isolationPFNeutrHadrCands() const;
    const std::vector<reco::PFCandidatePtr>& isolationPFGammaCands() const;

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
    float emFraction() const;        // Ecal/Hcal Cluster Energy
    float hcalTotOverPLead() const;  // total Hcal Cluster E / leadPFChargedHadron P
    float hcalMaxOverPLead() const;  // max. Hcal Cluster E / leadPFChargedHadron P
    // Hcal Cluster E in R<0.184 around Ecal impact point of leading track / leadPFChargedHadron P
    float hcal3x3OverPLead() const;
    float ecalStripSumEOverPLead() const;       // Simple BremsRecovery Sum E / leadPFChargedHadron P
    float bremsRecoveryEOverPLead() const;      // BremsRecovery Sum E / leadPFChargedHadron P
    reco::TrackRef electronPreIDTrack() const;  // Ref to KF track from Electron PreID
    float electronPreIDOutput() const;          // BDT output from Electron PreID
    bool electronPreIDDecision() const;         // Decision from Electron PreID

    void setemFraction(const float&);
    void sethcalTotOverPLead(const float&);
    void sethcalMaxOverPLead(const float&);
    void sethcal3x3OverPLead(const float&);
    void setecalStripSumEOverPLead(const float&);
    void setbremsRecoveryEOverPLead(const float&);
    void setelectronPreIDTrack(const reco::TrackRef&);
    void setelectronPreIDOutput(const float&);
    void setelectronPreIDDecision(const bool&);

    // For Muon Rejection
    bool hasMuonReference() const;  // check if muon ref exists
    float caloComp() const;
    float segComp() const;
    bool muonDecision() const;
    void setCaloComp(const float&);
    void setSegComp(const float&);
    void setMuonDecision(const bool&);

    /// return the number of source Candidates
    /// ( the candidates used to construct this Candidate)
    /// in the case of taus, there is only one source candidate,
    /// which is the corresponding PFJet
    size_type numberOfSourceCandidatePtrs() const override { return 1; }

    /// return a RefToBase to the source Candidates
    /// ( the candidates used to construct this Candidate)
    CandidatePtr sourceCandidatePtr(size_type i) const override;

    /// prints information on this PFTau
    void dump(std::ostream& out = std::cout) const;

  private:
    friend class tau::RecoTauConstructor;
    friend class tau::PFRecoTauEnergyAlgorithmPlugin;

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
    PFTauTagInfoRef PFTauTagInfoRef_;
    reco::CandidatePtr leadChargedHadrCand_;
    reco::CandidatePtr leadNeutralCand_;
    reco::CandidatePtr leadCand_;
    reco::TrackRef electronPreIDTrack_;

    // Signal candidates
    std::vector<reco::CandidatePtr> selectedSignalCands_;
    std::vector<reco::CandidatePtr> selectedSignalChargedHadrCands_;
    std::vector<reco::CandidatePtr> selectedSignalNeutrHadrCands_;
    std::vector<reco::CandidatePtr> selectedSignalGammaCands_;

    // Isolation candidates
    std::vector<reco::CandidatePtr> selectedIsolationCands_;
    std::vector<reco::CandidatePtr> selectedIsolationChargedHadrCands_;
    std::vector<reco::CandidatePtr> selectedIsolationNeutrHadrCands_;
    std::vector<reco::CandidatePtr> selectedIsolationGammaCands_;

    // Transient caches for PFCandidate-based accessors
    edm::AtomicPtrCache<reco::PFCandidatePtr> leadPFChargedHadrCand_;
    edm::AtomicPtrCache<reco::PFCandidatePtr> leadPFNeutralCand_;
    edm::AtomicPtrCache<reco::PFCandidatePtr> leadPFCand_;

    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientSignalPFCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientSignalPFChargedHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientSignalPFNeutrHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientSignalPFGammaCands_;

    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientIsolationPFCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientIsolationPFChargedHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientIsolationPFNeutrHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr>> selectedTransientIsolationPFGammaCands_;

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

  std::ostream& operator<<(std::ostream& out, const PFTau& c);

}  // end namespace reco

#endif
