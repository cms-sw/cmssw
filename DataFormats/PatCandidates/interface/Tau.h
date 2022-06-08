#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    pat::Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   pat::Tau implements the analysis-level tau class within the 'pat' namespace.
   It inherits from reco::BaseTau, copies all the information from the source
   reco::PFTau, and adds some PAT-specific variables.

   Please post comments and questions to the Physics Tools hypernews:
   https://hypernews.cern.ch/HyperNews/CMS/get/physTools.html

  \author Steven Lowette, Christophe Delaere, Giovanni Petrucciani, Frederic Ronga, Colin Bernet
*/

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"
#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"
#include "DataFormats/PatCandidates/interface/TauPFEssential.h"

#include "DataFormats/Common/interface/AtomicPtrCache.h"

// Define typedefs for convenience
namespace pat {
  class Tau;
  typedef std::vector<Tau> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  typedef edm::RefProd<TauCollection> TauRefProd;
  typedef edm::RefVector<TauCollection> TauRefVector;
}  // namespace pat

namespace reco {
  /// pipe operator (introduced to use pat::Tau with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Tau& obj);
  //  bool sortByPt(const CandidatePtrVector::const_iterator &lhs, const CandidatePtrVector::const_iterator &rhs) { return (*lhs)->pt() < (*rhs)->pt(); }
}  // namespace reco

// Class definition
namespace pat {

  class PATTauSlimmer;

  class Tau : public Lepton<reco::BaseTau> {
    /// make friends with PATTauProducer so that it can set the initial
    /// jet energy scale unequal to raw calling the private initializeJEC
    /// function, which should be non accessible to any other user
    friend class PATTauProducer;

  public:
    typedef std::pair<std::string, float> IdPair;

    /// default constructor
    Tau();
    /// constructor from a reco tau
    Tau(const reco::BaseTau& aTau);
    /// constructor from a RefToBase to a reco tau (to be superseded by Ptr counterpart)
    Tau(const edm::RefToBase<reco::BaseTau>& aTauRef);
    /// constructor from a Ptr to a reco tau
    Tau(const edm::Ptr<reco::BaseTau>& aTauRef);
    /// destructor
    ~Tau() override;

    /// required reimplementation of the Candidate's clone method
    Tau* clone() const override { return new Tau(*this); }

    // ---- methods for content embedding ----
    /// override the reco::BaseTau::isolationTracks method, to access the internal storage of the isolation tracks
    const reco::TrackRefVector& isolationTracks() const override;
    /// override the reco::BaseTau::leadTrack method, to access the internal storage of the leading track
    reco::TrackRef leadTrack() const override;
    /// override the reco::BaseTau::signalTracks method, to access the internal storage of the signal tracks
    const reco::TrackRefVector& signalTracks() const override;
    /// method to store the isolation tracks internally
    void embedIsolationTracks();
    /// method to store the leading track internally
    void embedLeadTrack();
    /// method to store the signal tracks internally
    void embedSignalTracks();
    ///- PFTau specific content -
    /// method to store the leading candidate internally
    void embedLeadPFCand();
    /// method to store the leading charged hadron candidate internally
    void embedLeadPFChargedHadrCand();
    /// method to store the leading neutral candidate internally
    void embedLeadPFNeutralCand();
    /// method to store the signal candidates internally
    void embedSignalPFCands();
    /// method to store the signal charged hadrons candidates internally
    void embedSignalPFChargedHadrCands();
    /// method to store the signal neutral hadrons candidates internally
    void embedSignalPFNeutralHadrCands();
    /// method to store the signal gamma candidates internally
    void embedSignalPFGammaCands();
    /// method to store the isolation candidates internally
    void embedIsolationPFCands();
    /// method to store the isolation charged hadrons candidates internally
    void embedIsolationPFChargedHadrCands();
    /// method to store the isolation neutral hadrons candidates internally
    void embedIsolationPFNeutralHadrCands();
    /// method to store the isolation gamma candidates internally
    void embedIsolationPFGammaCands();

    // ---- matched GenJet methods ----
    /// return matched GenJet, built from the visible particles of a generated tau
    const reco::GenJet* genJet() const;
    /// set the matched GenJet
    void setGenJet(const reco::GenJetRef& ref);

    // ---- PFTau accessors (getters only) ----
    /// Returns true if this pat::Tau was made from a reco::PFTau
    bool isPFTau() const { return !pfSpecific_.empty(); }
    /// return PFTau info or throw exception 'not PFTau'
    const pat::tau::TauPFSpecific& pfSpecific() const;
    const pat::tau::TauPFEssential& pfEssential() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const reco::JetBaseRef& pfJetRef() const { return pfSpecific().pfJetRef_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    reco::PFRecoTauChargedHadronRef leadTauChargedHadronCandidate() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const reco::PFCandidatePtr leadPFChargedHadrCand() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float leadPFChargedHadrCandsignedSipt() const { return pfSpecific().leadPFChargedHadrCandsignedSipt_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const reco::PFCandidatePtr leadPFNeutralCand() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const reco::PFCandidatePtr leadPFCand() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& signalPFCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& signalPFChargedHadrCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& signalPFNeutrHadrCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& signalPFGammaCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFRecoTauChargedHadron>& signalTauChargedHadronCandidates() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::RecoTauPiZero>& signalPiZeroCandidates() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& isolationPFCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& isolationPFChargedHadrCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& isolationPFNeutrHadrCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFCandidatePtr>& isolationPFGammaCands() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::PFRecoTauChargedHadron>& isolationTauChargedHadronCandidates() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const std::vector<reco::RecoTauPiZero>& isolationPiZeroCandidates() const;
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float isolationPFChargedHadrCandsPtSum() const { return pfSpecific().isolationPFChargedHadrCandsPtSum_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float isolationPFGammaCandsEtSum() const { return pfSpecific().isolationPFGammaCandsEtSum_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float maximumHCALPFClusterEt() const { return pfSpecific().maximumHCALPFClusterEt_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float emFraction() const { return pfSpecific().emFraction_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float hcalTotOverPLead() const { return pfSpecific().hcalTotOverPLead_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float hcalMaxOverPLead() const { return pfSpecific().hcalMaxOverPLead_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float hcal3x3OverPLead() const { return pfSpecific().hcal3x3OverPLead_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float ecalStripSumEOverPLead() const { return pfSpecific().ecalStripSumEOverPLead_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float bremsRecoveryEOverPLead() const { return pfSpecific().bremsRecoveryEOverPLead_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const reco::TrackRef& electronPreIDTrack() const { return pfSpecific().electronPreIDTrack_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float electronPreIDOutput() const { return pfSpecific().electronPreIDOutput_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    bool electronPreIDDecision() const { return pfSpecific().electronPreIDDecision_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float caloComp() const { return pfSpecific().caloComp_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    float segComp() const { return pfSpecific().segComp_; }
    /// Method copied from reco::PFTau.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    bool muonDecision() const { return pfSpecific().muonDecision_; }

    /// ----- Methods returning associated PFCandidates that work on PAT+AOD, PAT+embedding and miniAOD -----
    /// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
    const reco::CandidatePtr leadChargedHadrCand() const;
    /// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
    const reco::CandidatePtr leadNeutralCand() const;
    /// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
    const reco::CandidatePtr leadCand() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    bool ExistSignalCands() const;
    bool ExistIsolationCands() const;
    reco::CandidatePtrVector signalCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector signalChargedHadrCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector signalNeutrHadrCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector signalGammaCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector isolationCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector isolationChargedHadrCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector isolationNeutrHadrCands() const;
    /// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
    /// note that the vector is returned by value.
    reco::CandidatePtrVector isolationGammaCands() const;

    /// return the PackedCandidates on miniAOD corresponding with tau "lost" tracks
    /// note that the vector is returned by value.
    std::vector<reco::CandidatePtr> signalLostTracks() const;

    /// setters for the PtrVectors (for miniAOD)
    void setSignalChargedHadrCands(const reco::CandidatePtrVector& ptrs) { signalChargedHadrCandPtrs_ = ptrs; }
    void setSignalNeutralHadrCands(const reco::CandidatePtrVector& ptrs) { signalNeutralHadrCandPtrs_ = ptrs; }
    void setSignalGammaCands(const reco::CandidatePtrVector& ptrs) { signalGammaCandPtrs_ = ptrs; }
    void setIsolationChargedHadrCands(const reco::CandidatePtrVector& ptrs) { isolationChargedHadrCandPtrs_ = ptrs; }
    void setIsolationNeutralHadrCands(const reco::CandidatePtrVector& ptrs) { isolationNeutralHadrCandPtrs_ = ptrs; }
    void setIsolationGammaCands(const reco::CandidatePtrVector& ptrs) { isolationGammaCandPtrs_ = ptrs; }
    void setSignalLostTracks(const std::vector<reco::CandidatePtr>& ptrs);

    /// ----- Top Projection business -------
    /// get the number of non-null PFCandidates
    size_t numberOfSourceCandidatePtrs() const override;
    /// get the source candidate pointer with index i
    reco::CandidatePtr sourceCandidatePtr(size_type i) const override;

    /// ---- Tau lifetime information ----
    /// Filled from PFTauTIPAssociation.
    /// Throws an exception if this pat::Tau was not made from a reco::PFTau
    const pat::tau::TauPFEssential::Point& dxy_PCA() const { return pfEssential().dxy_PCA_; }
    float dxy() const { return pfEssential().dxy_; }
    float dxy_error() const { return pfEssential().dxy_error_; }
    float dxy_Sig() const;
    const reco::VertexRef& primaryVertex() const { return pfEssential().pv_; }
    const pat::tau::TauPFEssential::Point& primaryVertexPos() const { return pfEssential().pvPos_; }
    const pat::tau::TauPFEssential::CovMatrix& primaryVertexCov() const { return pfEssential().pvCov_; }
    bool hasSecondaryVertex() const { return pfEssential().hasSV_; }
    const pat::tau::TauPFEssential::Vector& flightLength() const { return pfEssential().flightLength_; }
    float flightLengthSig() const { return pfEssential().flightLengthSig_; }
    pat::tau::TauPFEssential::CovMatrix flightLengthCov() const;
    const reco::VertexRef& secondaryVertex() const { return pfEssential().sv_; }
    const pat::tau::TauPFEssential::Point& secondaryVertexPos() const { return pfEssential().svPos_; }
    const pat::tau::TauPFEssential::CovMatrix& secondaryVertexCov() const { return pfEssential().svCov_; }
    float ip3d() const { return pfEssential().ip3d_; }
    float ip3d_error() const { return pfEssential().ip3d_error_; }
    float ip3d_Sig() const;

    /// ---- Information for MVA isolation ----
    /// Needed to recompute MVA isolation on MiniAOD
    /// return sum of ecal energies from signal candidates
    float ecalEnergy() const { return pfEssential().ecalEnergy_; }
    /// return sum of hcal energies from signal candidates
    float hcalEnergy() const { return pfEssential().hcalEnergy_; }
    /// return normalized chi2 of leading track
    float leadingTrackNormChi2() const { return pfEssential().leadingTrackNormChi2_; }

    /// ---- Information for anti-electron training ----
    /// Needed to recompute on MiniAOD
    /// return ecal energy from LeadChargedHadrCand
    float ecalEnergyLeadChargedHadrCand() const { return pfEssential().ecalEnergyLeadChargedHadrCand_; }
    /// return hcal energy from LeadChargedHadrCand
    float hcalEnergyLeadChargedHadrCand() const { return pfEssential().hcalEnergyLeadChargedHadrCand_; }
    /// return phiAtEcalEntrance
    float phiAtEcalEntrance() const { return pfEssential().phiAtEcalEntrance_; }
    /// return etaAtEcalEntrance
    float etaAtEcalEntrance() const { return pfEssential().etaAtEcalEntrance_; }
    /// return etaAtEcalEntrance from LeadChargedCand
    float etaAtEcalEntranceLeadChargedCand() const { return pfEssential().etaAtEcalEntranceLeadChargedCand_; }
    /// return pt from  LeadChargedCand
    float ptLeadChargedCand() const { return pfEssential().ptLeadChargedCand_; }
    /// return emFraction_MVA
    float emFraction_MVA() const { return pfEssential().emFraction_; }

    /// Methods copied from reco::Jet.
    /// (accessible from reco::PFTau via reco::PFTauTagInfo)
    reco::Candidate::LorentzVector p4Jet() const;
    float etaetaMoment() const;
    float phiphiMoment() const;
    float etaphiMoment() const;

    /// reconstructed tau decay mode (specific to PFTau)
    int decayMode() const { return pfEssential().decayMode_; }
    /// set decay mode
    void setDecayMode(int);

    // ---- methods for tau ID ----
    /// Returns a specific tau ID associated to the pat::Tau given its name
    /// For cut-based IDs, the value is 1.0 for good, 0.0 for bad.
    /// The names are defined within the configuration parameterset "tauIDSources"
    /// in PhysicsTools/PatAlgos/python/producersLayer1/tauProducer_cfi.py .
    /// Note: an exception is thrown if the specified ID is not available
    float tauID(const std::string& name) const;
    float tauID(const char* name) const { return tauID(std::string(name)); }
    /// Returns true if a specific ID is available in this pat::Tau
    bool isTauIDAvailable(const std::string& name) const;
    /// Returns all the tau IDs in the form of <name,value> pairs
    /// The 'default' ID is the first in the list
    const std::vector<IdPair>& tauIDs() const { return tauIDs_; }
    /// Store multiple tau ID values, discarding existing ones
    /// The first one in the list becomes the 'default' tau id
    void setTauIDs(const std::vector<IdPair>& ids) { tauIDs_ = ids; }

    /// pipe operator (introduced to use pat::Tau with PFTopProjectors)
    friend std::ostream& reco::operator<<(std::ostream& out, const Tau& obj);

    /// ---- methods for jet corrections ----
    /// returns the labels of all available sets of jet energy corrections
    const std::vector<std::string> availableJECSets() const;
    // returns the available JEC Levels for a given jecSet
    const std::vector<std::string> availableJECLevels(const int& set = 0) const;
    // returns the available JEC Levels for a given jecSet
    const std::vector<std::string> availableJECLevels(const std::string& set) const {
      return availableJECLevels(jecSet(set));
    };
    /// returns true if the jet carries jet energy correction information
    /// at all
    bool jecSetsAvailable() const { return !jec_.empty(); }
    /// returns true if the jet carries a set of jet energy correction
    /// factors with the given label
    bool jecSetAvailable(const std::string& set) const { return (jecSet(set) >= 0); };
    /// returns true if the jet carries a set of jet energy correction
    /// factors with the given label
    bool jecSetAvailable(const unsigned int& set) const { return (set < jec_.size()); };
    /// returns the label of the current set of jet energy corrections
    std::string currentJECSet() const {
      return currentJECSet_ < jec_.size() ? jec_.at(currentJECSet_).jecSet() : std::string("ERROR");
    }
    /// return the name of the current step of jet energy corrections
    std::string currentJECLevel() const {
      return currentJECSet_ < jec_.size() ? jec_.at(currentJECSet_).jecLevel(currentJECLevel_) : std::string("ERROR");
    }
    /// correction factor to the given level for a specific set
    /// of correction factors, starting from the current level
    float jecFactor(const std::string& level, const std::string& set = "") const;
    /// correction factor to the given level for a specific set
    /// of correction factors, starting from the current level
    float jecFactor(const unsigned int& level, const unsigned int& set = 0) const;
    /// copy of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    Tau correctedTauJet(const std::string& level, const std::string& set = "") const;
    /// copy of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    Tau correctedTauJet(const unsigned int& level, const unsigned int& set = 0) const;
    /// p4 of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    const LorentzVector& correctedP4(const std::string& level, const std::string& set = "") const {
      return correctedTauJet(level, set).p4();
    }
    /// p4 of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    const LorentzVector& correctedP4(const unsigned int& level, const unsigned int& set = 0) const {
      return correctedTauJet(level, set).p4();
    }

    friend class PATTauSlimmer;

  protected:
    /// index of the set of jec factors with given label; returns -1 if no set
    /// of jec factors exists with the given label
    int jecSet(const std::string& label) const;
    /// update the current JEC set; used by correctedJet
    void currentJECSet(const unsigned int& set) { currentJECSet_ = set; };
    /// update the current JEC level; used by correctedJet
    void currentJECLevel(const unsigned int& level) { currentJECLevel_ = level; };
    /// add more sets of energy correction factors
    void addJECFactors(const TauJetCorrFactors& jec) { jec_.push_back(jec); };
    /// initialize the jet to a given JEC level during creation starting from Uncorrected
    void initializeJEC(unsigned int level, const unsigned int set = 0);

  private:
    /// helper to avoid code duplication in constructors
    void initFromBaseTau(const reco::BaseTau& aTau);
    // ---- for content embedding ----
    bool embeddedIsolationTracks_;
    std::vector<reco::Track> isolationTracks_;
    edm::AtomicPtrCache<reco::TrackRefVector> isolationTracksTransientRefVector_;
    bool embeddedLeadTrack_;
    std::vector<reco::Track> leadTrack_;
    bool embeddedSignalTracks_;
    std::vector<reco::Track> signalTracks_;
    edm::AtomicPtrCache<reco::TrackRefVector> signalTracksTransientRefVector_;
    // specific for PFTau
    std::vector<reco::PFCandidate> leadPFCand_;
    bool embeddedLeadPFCand_;
    std::vector<reco::PFCandidate> leadPFChargedHadrCand_;
    bool embeddedLeadPFChargedHadrCand_;
    std::vector<reco::PFCandidate> leadPFNeutralCand_;
    bool embeddedLeadPFNeutralCand_;

    std::vector<reco::PFCandidate> signalPFCands_;
    bool embeddedSignalPFCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > signalPFCandsTransientPtrs_;
    std::vector<reco::PFCandidate> signalPFChargedHadrCands_;
    bool embeddedSignalPFChargedHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > signalPFChargedHadrCandsTransientPtrs_;
    std::vector<reco::PFCandidate> signalPFNeutralHadrCands_;
    bool embeddedSignalPFNeutralHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > signalPFNeutralHadrCandsTransientPtrs_;
    std::vector<reco::PFCandidate> signalPFGammaCands_;
    bool embeddedSignalPFGammaCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > signalPFGammaCandsTransientPtrs_;
    std::vector<reco::PFCandidate> isolationPFCands_;
    bool embeddedIsolationPFCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > isolationPFCandsTransientPtrs_;
    std::vector<reco::PFCandidate> isolationPFChargedHadrCands_;
    bool embeddedIsolationPFChargedHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > isolationPFChargedHadrCandsTransientPtrs_;
    std::vector<reco::PFCandidate> isolationPFNeutralHadrCands_;
    bool embeddedIsolationPFNeutralHadrCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > isolationPFNeutralHadrCandsTransientPtrs_;
    std::vector<reco::PFCandidate> isolationPFGammaCands_;
    bool embeddedIsolationPFGammaCands_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > isolationPFGammaCandsTransientPtrs_;

    // ---- matched GenJet holder ----
    std::vector<reco::GenJet> genJet_;

    // ---- tau ID's holder ----
    std::vector<IdPair> tauIDs_;

    // ---- PFTau specific variables  ----
    /// holder for PFTau info, or empty vector if CaloTau
    std::vector<pat::tau::TauPFSpecific> pfSpecific_;

    // ---- energy scale correction factors ----
    // energy scale correction factors; the string carries a potential label if
    // more then one set of correction factors is embedded. The label corresponds
    // to the label of the jetCorrFactors module that has been embedded.
    std::vector<pat::TauJetCorrFactors> jec_;
    // currently applied set of jet energy correction factors (i.e. the index in
    // jetEnergyCorrections_)
    unsigned int currentJECSet_;
    // currently applied jet energy correction level
    unsigned int currentJECLevel_;

    // ---- references to packed pf candidates -----
    reco::CandidatePtrVector signalChargedHadrCandPtrs_;
    reco::CandidatePtrVector signalNeutralHadrCandPtrs_;
    reco::CandidatePtrVector signalGammaCandPtrs_;

    reco::CandidatePtrVector isolationChargedHadrCandPtrs_;
    reco::CandidatePtrVector isolationNeutralHadrCandPtrs_;
    reco::CandidatePtrVector isolationGammaCandPtrs_;

    // -- essential info to keep

    std::vector<pat::tau::TauPFEssential> pfEssential_;
  };
}  // namespace pat

#endif
