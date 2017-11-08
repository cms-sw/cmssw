#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//using namespace std;
namespace reco {

PFBaseTau::PFBaseTau() 
{
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

PFBaseTau::PFBaseTau(Charge q, const LorentzVector& p4, const Point& vtx) 
  : BaseTau(q, p4, vtx)
{
   leadPFChargedHadrCandsignedSipt_ = NAN;
   isolationPFChargedHadrCandsPtSum_ = NAN;
   isolationPFGammaCandsEtSum_ = NAN;
   maximumHCALPFClusterEt_ = NAN;

   emFraction_ = NAN;
   hcalTotOverPLead_ = NAN;
   hcalMaxOverPLead_ = NAN;
   hcal3x3OverPLead_ = NAN;
   ecalStripSumEOverPLead_= NAN;
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

PFBaseTau* PFBaseTau::clone() const { return new PFBaseTau(*this); }

// Constituent getters and setters
const JetBaseRef& PFBaseTau::jetRef() const { return jetRef_; }
void PFBaseTau::setjetRef(const JetBaseRef& x) { jetRef_ = x; }

// const PFTauTagInfoRef& PFBaseTau::PFTauTagInfoRef() const {
//   return PFTauTagInfoRef_;
// }

// void PFBaseTau::setPFTauTagInfoRef(const PFTauTagInfoRef x) { PFTauTagInfoRef_ = x; }

const CandidatePtr& PFBaseTau::leadPFChargedHadrCand() const { return leadPFChargedHadrCand_; }
const CandidatePtr& PFBaseTau::leadPFNeutralCand() const { return leadPFNeutralCand_; }
const CandidatePtr& PFBaseTau::leadPFCand() const { return leadPFCand_; }

void PFBaseTau::setleadPFChargedHadrCand(const CandidatePtr& myLead) { leadPFChargedHadrCand_ = myLead;}
void PFBaseTau::setleadPFNeutralCand(const CandidatePtr& myLead) { leadPFNeutralCand_ = myLead;}
void PFBaseTau::setleadPFCand(const CandidatePtr& myLead) { leadPFCand_ = myLead;}

float PFBaseTau::leadPFChargedHadrCandsignedSipt() const { return leadPFChargedHadrCandsignedSipt_; }
void PFBaseTau::setleadPFChargedHadrCandsignedSipt(const float& x){ leadPFChargedHadrCandsignedSipt_ = x; }

const std::vector<CandidatePtr>& PFBaseTau::signalPFCands() const { return selectedSignalPFCands_; }
void PFBaseTau::setsignalPFCands(const std::vector<CandidatePtr>& myParts)  { selectedSignalPFCands_ = myParts; }
const std::vector<CandidatePtr>& PFBaseTau::signalPFChargedHadrCands() const { return selectedSignalPFChargedHadrCands_; }
void PFBaseTau::setsignalPFChargedHadrCands(const std::vector<CandidatePtr>& myParts)  { selectedSignalPFChargedHadrCands_ = myParts; }
const std::vector<CandidatePtr>& PFBaseTau::signalPFNeutrHadrCands() const {return selectedSignalPFNeutrHadrCands_; }
void PFBaseTau::setsignalPFNeutrHadrCands(const std::vector<CandidatePtr>& myParts)  { selectedSignalPFNeutrHadrCands_ = myParts; }
const std::vector<CandidatePtr>& PFBaseTau::signalPFGammaCands() const { return selectedSignalPFGammaCands_; }
void PFBaseTau::setsignalPFGammaCands(const std::vector<CandidatePtr>& myParts)  { selectedSignalPFGammaCands_ = myParts; }

const std::vector<CandidatePtr>& PFBaseTau::isolationPFCands() const { return selectedIsolationPFCands_; }
void PFBaseTau::setisolationPFCands(const std::vector<CandidatePtr>& myParts)  { selectedIsolationPFCands_ = myParts;} 
const std::vector<CandidatePtr>& PFBaseTau::isolationPFChargedHadrCands() const { return selectedIsolationPFChargedHadrCands_; }
void PFBaseTau::setisolationPFChargedHadrCands(const std::vector<CandidatePtr>& myParts)  { selectedIsolationPFChargedHadrCands_ = myParts; }
const std::vector<CandidatePtr>& PFBaseTau::isolationPFNeutrHadrCands() const { return selectedIsolationPFNeutrHadrCands_; }
void PFBaseTau::setisolationPFNeutrHadrCands(const std::vector<CandidatePtr>& myParts)  { selectedIsolationPFNeutrHadrCands_ = myParts; }
const std::vector<CandidatePtr>& PFBaseTau::isolationPFGammaCands() const { return selectedIsolationPFGammaCands_; }
void PFBaseTau::setisolationPFGammaCands(const std::vector<CandidatePtr>& myParts)  { selectedIsolationPFGammaCands_ = myParts; }


namespace {
  template<typename T, typename U>
  void setCache( const T& iFrom, const edm::AtomicPtrCache<U>& oCache) {
    if ( not oCache.isSet()) {
      // Fill them from the refs
      auto temp = std::make_unique<U>();
      temp->reserve(iFrom.size());
      for ( auto const& ref: iFrom ) {
        temp->push_back(*ref);
      }
      oCache.set(std::move(temp));
    }
  }

  template<typename T>
  T& makeCacheIfNeeded(edm::AtomicPtrCache<T>& oCache) {
    if(not oCache.isSet()) {
      oCache.set(std::move(std::make_unique<T>()));
    }
    return *oCache;
  }

  template<typename T>
  void copyToCache(T&& iFrom, edm::AtomicPtrCache<T>& oCache) {
    oCache.reset();
    oCache.set( std::make_unique<T>(std::move(iFrom)));
  }
}

// PiZero and decay mode information
const std::vector<RecoTauPiZero>& PFBaseTau::signalPiZeroCandidates() const {
  // Check if the signal pi zeros are already filled
  setCache(signalPiZeroCandidatesRefs_, signalPiZeroCandidates_);
  return *signalPiZeroCandidates_;
}

std::vector<RecoTauPiZero>& PFBaseTau::signalPiZeroCandidatesRestricted() {
  // Check if the signal pi zeros are already filled
  return makeCacheIfNeeded(signalPiZeroCandidates_);
}

void PFBaseTau::setsignalPiZeroCandidates(std::vector<RecoTauPiZero> cands) {
   copyToCache(std::move(cands), signalPiZeroCandidates_);
}

void PFBaseTau::setSignalPiZeroCandidatesRefs(RecoTauPiZeroRefVector cands) {
  signalPiZeroCandidatesRefs_ = std::move(cands);
}

const std::vector<RecoTauPiZero>& PFBaseTau::isolationPiZeroCandidates() const {
  // Check if the signal pi zeros are already filled
  setCache(isolationPiZeroCandidatesRefs_, isolationPiZeroCandidates_);
  return *isolationPiZeroCandidates_;
}

std::vector<RecoTauPiZero>& PFBaseTau::isolationPiZeroCandidatesRestricted() {
  // Check if the signal pi zeros are already filled
  return makeCacheIfNeeded(isolationPiZeroCandidates_);
}

void PFBaseTau::setIsolationPiZeroCandidatesRefs(RecoTauPiZeroRefVector cands) {
  isolationPiZeroCandidatesRefs_ = std::move(cands);
}

void PFBaseTau::setisolationPiZeroCandidates(std::vector<RecoTauPiZero> cands) {
  copyToCache(std::move(cands), signalPiZeroCandidates_);
}

// Tau Charged Hadron information
PFRecoTauChargedHadronRef PFBaseTau::leadTauChargedHadronCandidate() const {
  if ( !signalTauChargedHadronCandidatesRefs_.empty() ) {
    return signalTauChargedHadronCandidatesRefs_[0];
  } else {
    return PFRecoTauChargedHadronRef();
  }
}

const std::vector<PFRecoTauChargedHadron>& PFBaseTau::signalTauChargedHadronCandidates() const {
  // Check if the signal tau charged hadrons are already filled
  setCache(signalTauChargedHadronCandidatesRefs_, signalTauChargedHadronCandidates_);
  return *signalTauChargedHadronCandidates_;
}

std::vector<PFRecoTauChargedHadron>& PFBaseTau::signalTauChargedHadronCandidatesRestricted() {
  // Check if the signal tau charged hadrons are already filled
  return makeCacheIfNeeded(signalTauChargedHadronCandidates_);
}

void PFBaseTau::setSignalTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron> cands) {
  copyToCache(std::move(cands), signalTauChargedHadronCandidates_);
}

void PFBaseTau::setSignalTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector cands) {
  signalTauChargedHadronCandidatesRefs_ = std::move(cands);
}

const std::vector<PFRecoTauChargedHadron>& PFBaseTau::isolationTauChargedHadronCandidates() const {
  // Check if the isolation tau charged hadrons are already filled
  setCache(isolationTauChargedHadronCandidatesRefs_, isolationTauChargedHadronCandidates_);
  return *isolationTauChargedHadronCandidates_;
}

std::vector<PFRecoTauChargedHadron>& PFBaseTau::isolationTauChargedHadronCandidatesRestricted() {
  // Check if the isolation tau charged hadrons are already filled
  return makeCacheIfNeeded(isolationTauChargedHadronCandidates_);
}

void PFBaseTau::setIsolationTauChargedHadronCandidates(std::vector<PFRecoTauChargedHadron> cands) {
  copyToCache(std::move(cands),isolationTauChargedHadronCandidates_);
}

void PFBaseTau::setIsolationTauChargedHadronCandidatesRefs(PFRecoTauChargedHadronRefVector cands) {
  isolationTauChargedHadronCandidatesRefs_ = std::move(cands);
}

PFBaseTau::hadronicDecayMode PFBaseTau::decayMode() const { return decayMode_; }

void PFBaseTau::setDecayMode(const PFBaseTau::hadronicDecayMode& dm){ decayMode_=dm;}

// Setting information about the isolation region
float PFBaseTau::isolationPFChargedHadrCandsPtSum() const {return isolationPFChargedHadrCandsPtSum_;}
void PFBaseTau::setisolationPFChargedHadrCandsPtSum(const float& x){isolationPFChargedHadrCandsPtSum_=x;}

float PFBaseTau::isolationPFGammaCandsEtSum() const {return isolationPFGammaCandsEtSum_;}
void PFBaseTau::setisolationPFGammaCandsEtSum(const float& x){isolationPFGammaCandsEtSum_=x;}

float PFBaseTau::maximumHCALPFClusterEt() const {return maximumHCALPFClusterEt_;}
void PFBaseTau::setmaximumHCALPFClusterEt(const float& x){maximumHCALPFClusterEt_=x;}

// Electron variables
float PFBaseTau::emFraction() const {return emFraction_;}
float PFBaseTau::hcalTotOverPLead() const {return hcalTotOverPLead_;}
float PFBaseTau::hcalMaxOverPLead() const {return hcalMaxOverPLead_;}
float PFBaseTau::hcal3x3OverPLead() const {return hcal3x3OverPLead_;}
float PFBaseTau::ecalStripSumEOverPLead() const {return ecalStripSumEOverPLead_;}
float PFBaseTau::bremsRecoveryEOverPLead() const {return bremsRecoveryEOverPLead_;}
reco::TrackRef PFBaseTau::electronPreIDTrack() const {return electronPreIDTrack_;}
float PFBaseTau::electronPreIDOutput() const {return electronPreIDOutput_;}
bool PFBaseTau::electronPreIDDecision() const {return electronPreIDDecision_;}

void PFBaseTau::setemFraction(const float& x) {emFraction_ = x;}
void PFBaseTau::sethcalTotOverPLead(const float& x) {hcalTotOverPLead_ = x;}
void PFBaseTau::sethcalMaxOverPLead(const float& x) {hcalMaxOverPLead_ = x;}
void PFBaseTau::sethcal3x3OverPLead(const float& x) {hcal3x3OverPLead_ = x;}
void PFBaseTau::setecalStripSumEOverPLead(const float& x) {ecalStripSumEOverPLead_ = x;}
void PFBaseTau::setbremsRecoveryEOverPLead(const float& x) {bremsRecoveryEOverPLead_ = x;}
void PFBaseTau::setelectronPreIDTrack(const reco::TrackRef& x) {electronPreIDTrack_ = x;}
void PFBaseTau::setelectronPreIDOutput(const float& x) {electronPreIDOutput_ = x;}
void PFBaseTau::setelectronPreIDDecision(const bool& x) {electronPreIDDecision_ = x;}

float PFBaseTau::caloComp() const {return caloComp_;}
float PFBaseTau::segComp() const {return segComp_;}
bool  PFBaseTau::muonDecision() const {return muonDecision_;}
void PFBaseTau::setCaloComp(const float& x) {caloComp_ = x;}
void PFBaseTau::setSegComp (const float& x) {segComp_  = x;}
void PFBaseTau::setMuonDecision(const bool& x) {muonDecision_ = x;}

CandidatePtr PFBaseTau::sourceCandidatePtr( size_type i ) const {
    if ( i!=0 ) return CandidatePtr();
    return jetRef().castTo<CandidatePtr>();
}


bool PFBaseTau::overlap(const Candidate& theCand) const {
    const RecoCandidate* theRecoCand = dynamic_cast<const RecoCandidate *>(&theCand);
    return (theRecoCand!=nullptr && (checkOverlap(track(), theRecoCand->track())));
}

void PFBaseTau::dump(std::ostream& out) const {

    if(!out) return;
    
    out<<"Pt of the PFBaseTau "<<pt()<<std::endl;
    const CandidatePtr& theLeadPFCand = leadPFChargedHadrCand();
    if(!theLeadPFCand){
        out<<"No Lead PFCand "<<std::endl;
    }else{
        out<<"Lead PFCand PDG Id " << (*theLeadPFCand).pdgId() << std::endl;
        out<<"Lead PFCand Pt "<<(*theLeadPFCand).pt()<<std::endl;
        out<<"Lead PFCand Charge "<<(*theLeadPFCand).charge()<<std::endl;
        out<<"Inner point position (x,y,z) of the PFBaseTau ("<<vx()<<","<<vy()<<","<<vz()<<")"<<std::endl;
        out<<"Charge of the PFBaseTau "<<charge()<<std::endl;
        out<<"Et of the highest Et HCAL PFCluster "<<maximumHCALPFClusterEt()<<std::endl;
        out<<"Number of SignalPFChargedHadrCands = "<<signalPFChargedHadrCands().size()<<std::endl;
        out<<"Number of SignalPFGammaCands = "<<signalPFGammaCands().size()<<std::endl;
        out<<"Number of IsolationPFChargedHadrCands = "<<isolationPFChargedHadrCands().size()<<std::endl;
        out<<"Number of IsolationPFGammaCands = "<<isolationPFGammaCands().size()<<std::endl;
        out<<"Sum of Pt of charged hadr. Candidates in isolation annulus around Lead PF = "<<isolationPFChargedHadrCandsPtSum()<<std::endl;
        out<<"Sum of Et of gamma Candidates in other isolation annulus around Lead PF = "<<isolationPFGammaCandsEtSum()<<std::endl;

    }
    // return out;
}

std::ostream& operator<<(std::ostream& out, const reco::PFBaseTau& tau) {

   if(!out) return out;

   out << std::setprecision(3)
     <<"PFBaseTau "
     << " charge: " << tau.charge() << " "
     << " pt:" <<tau.pt()<<" "
     << " eta:" <<tau.eta()<<" "
     << " phi:" <<tau.phi()<<" "
     << " mass:" << tau.mass() << " "
     << " dm: " << tau.decayMode() << " "
     <<tau.signalPFCands().size()<<","
     <<tau.signalPFChargedHadrCands().size()<<","
     <<tau.signalPFGammaCands().size()<<","
     <<tau.signalPiZeroCandidates().size()<<","
     <<tau.signalPFNeutrHadrCands().size()<<"  "

     <<tau.isolationPFCands().size()<<","
     <<tau.isolationPFChargedHadrCands().size()<<","
     <<tau.isolationPFGammaCands().size()<<","
     <<tau.isolationPiZeroCandidates().size()<<","
     <<tau.isolationPFNeutrHadrCands().size();

   return out;
}

}
