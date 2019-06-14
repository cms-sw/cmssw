#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//using namespace std;
namespace reco {

  PFTau::PFTau() {
    leadPFChargedHadrCandsignedSipt_ = NAN;
    isolationPFChargedHadrCandsPtSum_ = NAN;
    isolationPFGammaCandsEtSum_ = NAN;
    maximumHCALPFClusterEt_ = NAN;
    emFraction_ = NAN;
    hcalTotOverPLead_ = NAN;
    hcalMaxOverPLead_ = NAN;
    hcal3x3OverPLead_ = NAN;
    ecalStripSumEOverPLead_ = NAN;
    bremsRecoveryEOverPLead_ = NAN;
    electronPreIDOutput_ = NAN;
    electronPreIDDecision_ = NAN;
    caloComp_ = NAN;
    segComp_ = NAN;
    muonDecision_ = NAN;
    decayMode_ = kNull;
    bendCorrMass_ = 0.;
    signalConeSize_ = 0.;
  }

  PFTau::PFTau(Charge q, const LorentzVector& p4, const Point& vtx) : BaseTau(q, p4, vtx) {
    leadPFChargedHadrCandsignedSipt_ = NAN;
    isolationPFChargedHadrCandsPtSum_ = NAN;
    isolationPFGammaCandsEtSum_ = NAN;
    maximumHCALPFClusterEt_ = NAN;

    emFraction_ = NAN;
    hcalTotOverPLead_ = NAN;
    hcalMaxOverPLead_ = NAN;
    hcal3x3OverPLead_ = NAN;
    ecalStripSumEOverPLead_ = NAN;
    bremsRecoveryEOverPLead_ = NAN;
    electronPreIDOutput_ = NAN;
    electronPreIDDecision_ = NAN;

    caloComp_ = NAN;
    segComp_ = NAN;
    muonDecision_ = NAN;
    decayMode_ = kNull;
    bendCorrMass_ = 0.;
    signalConeSize_ = 0.;
  }

  PFTau* PFTau::clone() const { return new PFTau(*this); }

  // Constituent getters and setters
  const JetBaseRef& PFTau::jetRef() const { return jetRef_; }
  void PFTau::setjetRef(const JetBaseRef& x) { jetRef_ = x; }

  const PFTauTagInfoRef& PFTau::pfTauTagInfoRef() const { return PFTauTagInfoRef_; }

  void PFTau::setpfTauTagInfoRef(const PFTauTagInfoRef x) { PFTauTagInfoRef_ = x; }

  const CandidatePtr& PFTau::leadChargedHadrCand() const { return leadChargedHadrCand_; }
  const CandidatePtr& PFTau::leadNeutralCand() const { return leadNeutralCand_; }
  const CandidatePtr& PFTau::leadCand() const { return leadCand_; }

  void PFTau::setleadChargedHadrCand(const CandidatePtr& myLead) { leadChargedHadrCand_ = myLead; }
  void PFTau::setleadNeutralCand(const CandidatePtr& myLead) { leadNeutralCand_ = myLead; }
  void PFTau::setleadCand(const CandidatePtr& myLead) { leadCand_ = myLead; }

  float PFTau::leadPFChargedHadrCandsignedSipt() const { return leadPFChargedHadrCandsignedSipt_; }
  void PFTau::setleadPFChargedHadrCandsignedSipt(const float& x) { leadPFChargedHadrCandsignedSipt_ = x; }

  const std::vector<CandidatePtr>& PFTau::signalCands() const { return selectedSignalCands_; }
  void PFTau::setsignalCands(const std::vector<CandidatePtr>& myParts) { selectedSignalCands_ = myParts; }
  const std::vector<CandidatePtr>& PFTau::signalChargedHadrCands() const { return selectedSignalChargedHadrCands_; }
  void PFTau::setsignalChargedHadrCands(const std::vector<CandidatePtr>& myParts) {
    selectedSignalChargedHadrCands_ = myParts;
  }
  const std::vector<CandidatePtr>& PFTau::signalNeutrHadrCands() const { return selectedSignalNeutrHadrCands_; }
  void PFTau::setsignalNeutrHadrCands(const std::vector<CandidatePtr>& myParts) {
    selectedSignalNeutrHadrCands_ = myParts;
  }
  const std::vector<CandidatePtr>& PFTau::signalGammaCands() const { return selectedSignalGammaCands_; }
  void PFTau::setsignalGammaCands(const std::vector<CandidatePtr>& myParts) { selectedSignalGammaCands_ = myParts; }

  const std::vector<CandidatePtr>& PFTau::isolationCands() const { return selectedIsolationCands_; }
  void PFTau::setisolationCands(const std::vector<CandidatePtr>& myParts) { selectedIsolationCands_ = myParts; }
  const std::vector<CandidatePtr>& PFTau::isolationChargedHadrCands() const {
    return selectedIsolationChargedHadrCands_;
  }
  void PFTau::setisolationChargedHadrCands(const std::vector<CandidatePtr>& myParts) {
    selectedIsolationChargedHadrCands_ = myParts;
  }
  const std::vector<CandidatePtr>& PFTau::isolationNeutrHadrCands() const { return selectedIsolationNeutrHadrCands_; }
  void PFTau::setisolationNeutrHadrCands(const std::vector<CandidatePtr>& myParts) {
    selectedIsolationNeutrHadrCands_ = myParts;
  }
  const std::vector<CandidatePtr>& PFTau::isolationGammaCands() const { return selectedIsolationGammaCands_; }
  void PFTau::setisolationGammaCands(const std::vector<CandidatePtr>& myParts) {
    selectedIsolationGammaCands_ = myParts;
  }

  namespace {
    template <typename T, typename U>
    void setCache(const T& iFrom, const edm::AtomicPtrCache<U>& oCache) {
      if (not oCache.isSet()) {
        // Fill them from the refs
        auto temp = std::make_unique<U>();
        temp->reserve(iFrom.size());
        for (auto const& ref : iFrom) {
          temp->push_back(*ref);
        }
        oCache.set(std::move(temp));
      }
    }

    template <typename T>
    T& makeCacheIfNeeded(edm::AtomicPtrCache<T>& oCache) {
      if (not oCache.isSet()) {
        oCache.set(std::move(std::make_unique<T>()));
      }
      return *oCache;
    }

    template <typename T>
    void copyToCache(T&& iFrom, edm::AtomicPtrCache<T>& oCache) {
      oCache.reset();
      oCache.set(std::make_unique<T>(std::move(iFrom)));
    }

    std::unique_ptr<reco::PFCandidatePtr> convertToPFPtr(const reco::CandidatePtr& ptr) {
      if (ptr.isNonnull()) {
        const reco::PFCandidate* pf_cand = dynamic_cast<const reco::PFCandidate*>(&*ptr);
        if (pf_cand != nullptr) {
          return std::unique_ptr<reco::PFCandidatePtr>(new reco::PFCandidatePtr(ptr));
        } else
          throw cms::Exception("Type Mismatch")
              << "This PFTau was not made from PFCandidates, but it is being tried to access a PFCandidate.\n";
      }
      return std::unique_ptr<reco::PFCandidatePtr>(new reco::PFCandidatePtr());
    }

    std::unique_ptr<std::vector<reco::PFCandidatePtr> > convertToPFPtrs(const std::vector<reco::CandidatePtr>& cands) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > newSignalPFCands{new std::vector<reco::PFCandidatePtr>{}};
      bool isPF = false;
      for (auto& cand : cands) {
        // Check for first Candidate if it is a PFCandidate; if yes, skip for the rest
        if (!isPF) {
          const reco::PFCandidate* pf_cand = dynamic_cast<const reco::PFCandidate*>(&*cand);
          if (pf_cand != nullptr) {
            isPF = true;
            newSignalPFCands->reserve(cands.size());
          } else
            throw cms::Exception("Type Mismatch")
                << "This PFTau was not made from PFCandidates, but it is being tried to access PFCandidates.\n";
        }
        const auto& newPtr = edm::Ptr<reco::PFCandidate>(cand);
        newSignalPFCands->push_back(newPtr);
      }
      return newSignalPFCands;
    }
  }  // namespace

  const PFCandidatePtr PFTau::leadPFChargedHadrCand() const {
    if (!leadPFChargedHadrCand_.isSet())
      leadPFChargedHadrCand_.set(convertToPFPtr(leadChargedHadrCand_));
    return *leadPFChargedHadrCand_;
  }

  const PFCandidatePtr PFTau::leadPFNeutralCand() const {
    if (!leadPFNeutralCand_.isSet())
      leadPFNeutralCand_.set(convertToPFPtr(leadNeutralCand_));
    return *leadPFNeutralCand_;
  }

  const PFCandidatePtr PFTau::leadPFCand() const {
    if (!leadPFCand_.isSet())
      leadPFCand_.set(convertToPFPtr(leadCand_));
    return *leadPFCand_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::signalPFCands() const {
    if (!selectedTransientSignalPFCands_.isSet()) {
      selectedTransientSignalPFCands_.set(convertToPFPtrs(selectedSignalCands_));
    }
    return *selectedTransientSignalPFCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::signalPFChargedHadrCands() const {
    if (!selectedTransientSignalPFChargedHadrCands_.isSet()) {
      selectedTransientSignalPFChargedHadrCands_.set(convertToPFPtrs(selectedSignalChargedHadrCands_));
    }
    return *selectedTransientSignalPFChargedHadrCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::signalPFNeutrHadrCands() const {
    if (!selectedTransientSignalPFNeutrHadrCands_.isSet()) {
      selectedTransientSignalPFNeutrHadrCands_.set(convertToPFPtrs(selectedSignalNeutrHadrCands_));
    }
    return *selectedTransientSignalPFNeutrHadrCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::signalPFGammaCands() const {
    if (!selectedTransientSignalPFGammaCands_.isSet()) {
      selectedTransientSignalPFGammaCands_.set(convertToPFPtrs(selectedSignalGammaCands_));
    }
    return *selectedTransientSignalPFGammaCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::isolationPFCands() const {
    if (!selectedTransientIsolationPFCands_.isSet()) {
      selectedTransientIsolationPFCands_.set(convertToPFPtrs(selectedIsolationCands_));
    }
    return *selectedTransientIsolationPFCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::isolationPFChargedHadrCands() const {
    if (!selectedTransientIsolationPFChargedHadrCands_.isSet()) {
      selectedTransientIsolationPFChargedHadrCands_.set(convertToPFPtrs(selectedIsolationChargedHadrCands_));
    }
    return *selectedTransientIsolationPFChargedHadrCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::isolationPFNeutrHadrCands() const {
    if (!selectedTransientIsolationPFNeutrHadrCands_.isSet()) {
      selectedTransientIsolationPFNeutrHadrCands_.set(convertToPFPtrs(selectedIsolationNeutrHadrCands_));
    }
    return *selectedTransientIsolationPFNeutrHadrCands_;
  }

  const std::vector<reco::PFCandidatePtr>& PFTau::isolationPFGammaCands() const {
    if (!selectedTransientIsolationPFGammaCands_.isSet()) {
      selectedTransientIsolationPFGammaCands_.set(convertToPFPtrs(selectedIsolationGammaCands_));
    }
    return *selectedTransientIsolationPFGammaCands_;
  }

  // PiZero and decay mode information
  const std::vector<RecoTauPiZero>& PFTau::signalPiZeroCandidates() const {
    // Check if the signal pi zeros are already filled
    setCache(signalPiZeroCandidatesRefs_, signalPiZeroCandidates_);
    return *signalPiZeroCandidates_;
  }

  std::vector<RecoTauPiZero>& PFTau::signalPiZeroCandidatesRestricted() {
    // Check if the signal pi zeros are already filled
    return makeCacheIfNeeded(signalPiZeroCandidates_);
  }

  void PFTau::setsignalPiZeroCandidates(std::vector<RecoTauPiZero> cands) {
    copyToCache(std::move(cands), signalPiZeroCandidates_);
  }

  void PFTau::setSignalPiZeroCandidatesRefs(RecoTauPiZeroRefVector cands) {
    signalPiZeroCandidatesRefs_ = std::move(cands);
  }

  const std::vector<RecoTauPiZero>& PFTau::isolationPiZeroCandidates() const {
    // Check if the signal pi zeros are already filled
    setCache(isolationPiZeroCandidatesRefs_, isolationPiZeroCandidates_);
    return *isolationPiZeroCandidates_;
  }

  std::vector<RecoTauPiZero>& PFTau::isolationPiZeroCandidatesRestricted() {
    // Check if the signal pi zeros are already filled
    return makeCacheIfNeeded(isolationPiZeroCandidates_);
  }

  void PFTau::setIsolationPiZeroCandidatesRefs(RecoTauPiZeroRefVector cands) {
    isolationPiZeroCandidatesRefs_ = std::move(cands);
  }

  void PFTau::setisolationPiZeroCandidates(std::vector<RecoTauPiZero> cands) {
    copyToCache(std::move(cands), signalPiZeroCandidates_);
  }

  // Tau Charged Hadron information
  PFRecoTauChargedHadronRef PFTau::leadTauChargedHadronCandidate() const {
    if (!signalTauChargedHadronCandidatesRefs_.empty()) {
      return signalTauChargedHadronCandidatesRefs_[0];
    } else {
      return PFRecoTauChargedHadronRef();
    }
  }

  const std::vector<PFRecoTauChargedHadron>& PFTau::signalTauChargedHadronCandidates() const {
    // Check if the signal tau charged hadrons are already filled
    setCache(signalTauChargedHadronCandidatesRefs_, signalTauChargedHadronCandidates_);
    return *signalTauChargedHadronCandidates_;
  }

  std::vector<PFRecoTauChargedHadron>& PFTau::signalTauChargedHadronCandidatesRestricted() {
    // Check if the signal tau charged hadrons are already filled
    return makeCacheIfNeeded(signalTauChargedHadronCandidates_);
  }

  void PFTau::setSignalTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron> cands) {
    copyToCache(std::move(cands), signalTauChargedHadronCandidates_);
  }

  void PFTau::setSignalTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector cands) {
    signalTauChargedHadronCandidatesRefs_ = std::move(cands);
  }

  const std::vector<PFRecoTauChargedHadron>& PFTau::isolationTauChargedHadronCandidates() const {
    // Check if the isolation tau charged hadrons are already filled
    setCache(isolationTauChargedHadronCandidatesRefs_, isolationTauChargedHadronCandidates_);
    return *isolationTauChargedHadronCandidates_;
  }

  std::vector<PFRecoTauChargedHadron>& PFTau::isolationTauChargedHadronCandidatesRestricted() {
    // Check if the isolation tau charged hadrons are already filled
    return makeCacheIfNeeded(isolationTauChargedHadronCandidates_);
  }

  void PFTau::setIsolationTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron> cands) {
    copyToCache(std::move(cands), isolationTauChargedHadronCandidates_);
  }

  void PFTau::setIsolationTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector cands) {
    isolationTauChargedHadronCandidatesRefs_ = std::move(cands);
  }

  PFTau::hadronicDecayMode PFTau::decayMode() const { return decayMode_; }

  void PFTau::setDecayMode(const PFTau::hadronicDecayMode& dm) { decayMode_ = dm; }

  // Setting information about the isolation region
  float PFTau::isolationPFChargedHadrCandsPtSum() const { return isolationPFChargedHadrCandsPtSum_; }
  void PFTau::setisolationPFChargedHadrCandsPtSum(const float& x) { isolationPFChargedHadrCandsPtSum_ = x; }

  float PFTau::isolationPFGammaCandsEtSum() const { return isolationPFGammaCandsEtSum_; }
  void PFTau::setisolationPFGammaCandsEtSum(const float& x) { isolationPFGammaCandsEtSum_ = x; }

  float PFTau::maximumHCALPFClusterEt() const { return maximumHCALPFClusterEt_; }
  void PFTau::setmaximumHCALPFClusterEt(const float& x) { maximumHCALPFClusterEt_ = x; }

  // Electron variables
  float PFTau::emFraction() const { return emFraction_; }
  float PFTau::hcalTotOverPLead() const { return hcalTotOverPLead_; }
  float PFTau::hcalMaxOverPLead() const { return hcalMaxOverPLead_; }
  float PFTau::hcal3x3OverPLead() const { return hcal3x3OverPLead_; }
  float PFTau::ecalStripSumEOverPLead() const { return ecalStripSumEOverPLead_; }
  float PFTau::bremsRecoveryEOverPLead() const { return bremsRecoveryEOverPLead_; }
  reco::TrackRef PFTau::electronPreIDTrack() const { return electronPreIDTrack_; }
  float PFTau::electronPreIDOutput() const { return electronPreIDOutput_; }
  bool PFTau::electronPreIDDecision() const { return electronPreIDDecision_; }

  void PFTau::setemFraction(const float& x) { emFraction_ = x; }
  void PFTau::sethcalTotOverPLead(const float& x) { hcalTotOverPLead_ = x; }
  void PFTau::sethcalMaxOverPLead(const float& x) { hcalMaxOverPLead_ = x; }
  void PFTau::sethcal3x3OverPLead(const float& x) { hcal3x3OverPLead_ = x; }
  void PFTau::setecalStripSumEOverPLead(const float& x) { ecalStripSumEOverPLead_ = x; }
  void PFTau::setbremsRecoveryEOverPLead(const float& x) { bremsRecoveryEOverPLead_ = x; }
  void PFTau::setelectronPreIDTrack(const reco::TrackRef& x) { electronPreIDTrack_ = x; }
  void PFTau::setelectronPreIDOutput(const float& x) { electronPreIDOutput_ = x; }
  void PFTau::setelectronPreIDDecision(const bool& x) { electronPreIDDecision_ = x; }

  // Muon variables
  bool PFTau::hasMuonReference() const {  // check if muon ref exists
    if (leadChargedHadrCand_.isNull())
      return false;
    else if (leadChargedHadrCand_.isNonnull()) {
      const reco::PFCandidate* pf_cand = dynamic_cast<const reco::PFCandidate*>(&*leadChargedHadrCand_);
      if (pf_cand) {
        reco::MuonRef muonRef = pf_cand->muonRef();
        if (muonRef.isNull())
          return false;
        else if (muonRef.isNonnull())
          return true;
      }
    }
    return false;
  }

  float PFTau::caloComp() const { return caloComp_; }
  float PFTau::segComp() const { return segComp_; }
  bool PFTau::muonDecision() const { return muonDecision_; }
  void PFTau::setCaloComp(const float& x) { caloComp_ = x; }
  void PFTau::setSegComp(const float& x) { segComp_ = x; }
  void PFTau::setMuonDecision(const bool& x) { muonDecision_ = x; }

  CandidatePtr PFTau::sourceCandidatePtr(size_type i) const {
    if (i != 0)
      return CandidatePtr();
    return jetRef().castTo<CandidatePtr>();
  }

  bool PFTau::overlap(const Candidate& theCand) const {
    const RecoCandidate* theRecoCand = dynamic_cast<const RecoCandidate*>(&theCand);
    return (theRecoCand != nullptr && (checkOverlap(track(), theRecoCand->track())));
  }

  void PFTau::dump(std::ostream& out) const {
    if (!out)
      return;

    if (pfTauTagInfoRef().isNonnull()) {
      out << "Its TauTagInfo constituents :" << std::endl;
      out << "# Tracks " << pfTauTagInfoRef()->Tracks().size() << std::endl;
      out << "# PF charged hadr. cand's " << pfTauTagInfoRef()->PFChargedHadrCands().size() << std::endl;
      out << "# PF neutral hadr. cand's " << pfTauTagInfoRef()->PFNeutrHadrCands().size() << std::endl;
      out << "# PF gamma cand's " << pfTauTagInfoRef()->PFGammaCands().size() << std::endl;
    }
    out << "in detail :" << std::endl;

    out << "Pt of the PFTau " << pt() << std::endl;
    const CandidatePtr& theLeadCand = leadChargedHadrCand();
    if (!theLeadCand) {
      out << "No Lead Cand " << std::endl;
    } else {
      out << "Lead Cand PDG Id " << (*theLeadCand).pdgId() << std::endl;
      out << "Lead Cand Pt " << (*theLeadCand).pt() << std::endl;
      out << "Lead Cand Charge " << (*theLeadCand).charge() << std::endl;
      out << "Inner point position (x,y,z) of the PFTau (" << vx() << "," << vy() << "," << vz() << ")" << std::endl;
      out << "Charge of the PFTau " << charge() << std::endl;
      out << "Et of the highest Et HCAL PFCluster " << maximumHCALPFClusterEt() << std::endl;
      out << "Number of SignalChargedHadrCands = " << signalChargedHadrCands().size() << std::endl;
      out << "Number of SignalGammaCands = " << signalGammaCands().size() << std::endl;
      out << "Number of IsolationChargedHadrCands = " << isolationChargedHadrCands().size() << std::endl;
      out << "Number of IsolationGammaCands = " << isolationGammaCands().size() << std::endl;
      out << "Sum of Pt of charged hadr. PFCandidates in isolation annulus around Lead PF = "
          << isolationPFChargedHadrCandsPtSum() << std::endl;
      out << "Sum of Et of gamma PFCandidates in other isolation annulus around Lead PF = "
          << isolationPFGammaCandsEtSum() << std::endl;
    }
    // return out;
  }

  std::ostream& operator<<(std::ostream& out, const reco::PFTau& tau) {
    if (!out)
      return out;

    out << std::setprecision(3) << "PFTau "
        << " charge: " << tau.charge() << " "
        << " pt:" << tau.pt() << " "
        << " eta:" << tau.eta() << " "
        << " phi:" << tau.phi() << " "
        << " mass:" << tau.mass() << " "
        << " dm: " << tau.decayMode() << " " << tau.signalCands().size() << "," << tau.signalChargedHadrCands().size()
        << "," << tau.signalGammaCands().size() << "," << tau.signalPiZeroCandidates().size() << ","
        << tau.signalNeutrHadrCands().size() << "  "

        << tau.isolationCands().size() << "," << tau.isolationChargedHadrCands().size() << ","
        << tau.isolationGammaCands().size() << "," << tau.isolationPiZeroCandidates().size() << ","
        << tau.isolationNeutrHadrCands().size();

    return out;
  }

}  // namespace reco
