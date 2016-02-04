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
#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include <iostream>
#include <limits>


namespace reco {

namespace tau {
// Forward declaration
class RecoTauConstructor;
}

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
    PFTau(Charge q,const LorentzVector &,const Point & = Point( 0, 0, 0 ) );
    virtual ~PFTau() {};
    PFTau* clone() const;

    const PFJetRef& jetRef() const;
    void setjetRef(const PFJetRef&);

    // DEPRECATED functions to access the PFTauTagInfoRef
    const PFTauTagInfoRef& pfTauTagInfoRef() const;
    void setpfTauTagInfoRef(const PFTauTagInfoRef);

    const PFCandidateRef& leadPFChargedHadrCand() const;
    const PFCandidateRef& leadPFNeutralCand() const;
    //Can be either the charged or the neutral one
    const PFCandidateRef& leadPFCand() const;

    void setleadPFChargedHadrCand(const PFCandidateRef&);
    void setleadPFNeutralCand(const PFCandidateRef&);
    void setleadPFCand(const PFCandidateRef&);

    /// Signed transverse impact parameter significance of the Track
    /// associated to the leading charged PFCandidate
    float leadPFChargedHadrCandsignedSipt() const;
    void setleadPFChargedHadrCandsignedSipt(const float&);

    /// PFCandidates in signal region
    const PFCandidateRefVector& signalPFCands() const;
    void setsignalPFCands(const PFCandidateRefVector&);

    /// Charged hadrons in signal region
    const PFCandidateRefVector& signalPFChargedHadrCands() const;
    void setsignalPFChargedHadrCands(const PFCandidateRefVector&);

    /// Neutral hadrons in signal region
    const PFCandidateRefVector& signalPFNeutrHadrCands() const;
    void setsignalPFNeutrHadrCands(const PFCandidateRefVector&);

    /// Gamma candidates in signal region
    const PFCandidateRefVector& signalPFGammaCands() const;
    void setsignalPFGammaCands(const PFCandidateRefVector&);

    /// PFCandidates in isolation region
    const PFCandidateRefVector& isolationPFCands() const;
    void setisolationPFCands(const PFCandidateRefVector&);

    /// Charged candidates in isolation region
    const PFCandidateRefVector& isolationPFChargedHadrCands() const;
    void setisolationPFChargedHadrCands(const PFCandidateRefVector&);

    //// Neutral hadrons in isolation region
    const PFCandidateRefVector& isolationPFNeutrHadrCands() const;
    void setisolationPFNeutrHadrCands(const PFCandidateRefVector&);

    /// Gamma candidates in isolation region
    const PFCandidateRefVector& isolationPFGammaCands() const;
    void setisolationPFGammaCands(const PFCandidateRefVector&);

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
    void setsignalPiZeroCandidates(const std::vector<RecoTauPiZero>&);

    /// Retrieve the association of isolation region gamma candidates into candidate PiZeros
    const std::vector<RecoTauPiZero>& isolationPiZeroCandidates() const;
    void setisolationPiZeroCandidates(const std::vector<RecoTauPiZero>&);

    /// Retrieve the identified hadronic decay mode according to the number of
    /// charged and piZero candidates in the signal cone
    hadronicDecayMode decayMode() const;

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

    // For Muon Rejection
    bool hasMuonReference() const; // check if muon ref exists
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
    size_type numberOfSourceCandidatePtrs() const {return 1;}

    /// return a RefToBase to the source Candidates
    /// ( the candidates used to construct this Candidate)
    CandidatePtr sourceCandidatePtr( size_type i ) const;

    /// prints information on this PFTau
    void dump(std::ostream& out=std::cout) const;

  private:
    friend class reco::tau::RecoTauConstructor;
    // check overlap with another candidate
    virtual bool overlap(const Candidate&) const;

    reco::PFJetRef jetRef_;

    PFTauTagInfoRef PFTauTagInfoRef_;
    PFCandidateRef leadPFChargedHadrCand_;
    PFCandidateRef leadPFNeutralCand_, leadPFCand_;

    // SIP
    float leadPFChargedHadrCandsignedSipt_;

    // Signal candidates
    PFCandidateRefVector selectedSignalPFCands_,
                         selectedSignalPFChargedHadrCands_,
                         selectedSignalPFNeutrHadrCands_,
                         selectedSignalPFGammaCands_;

    // Isolation candidates
    PFCandidateRefVector selectedIsolationPFCands_,
                         selectedIsolationPFChargedHadrCands_,
                         selectedIsolationPFNeutrHadrCands_,
                         selectedIsolationPFGammaCands_;

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
    reco::TrackRef electronPreIDTrack_;
    float electronPreIDOutput_;
    bool electronPreIDDecision_;

    // Muon rejection variables
    float caloComp_;
    float segComp_;
    bool muonDecision_;

    // Association of gamma candidates into PiZeros
    std::vector<reco::RecoTauPiZero> signalPiZeroCandidates_;
    std::vector<reco::RecoTauPiZero> isolationPiZeroCandidates_;
};

std::ostream & operator<<(std::ostream& out, const PFTau& c);

}
#endif
