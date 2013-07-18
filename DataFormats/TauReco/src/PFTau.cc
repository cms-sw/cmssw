#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//using namespace std;
namespace reco {

PFTau::PFTau() 
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
    electronPreIDDecision_= NAN;
    caloComp_ = NAN;
    segComp_ = NAN;
    muonDecision_ = NAN;
}

PFTau::PFTau(Charge q, const LorentzVector& p4, const Point& vtx) 
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
   electronPreIDDecision_= NAN;

   caloComp_ = NAN;
   segComp_ = NAN;
   muonDecision_ = NAN;
}

PFTau* PFTau::clone() const { return new PFTau(*this); }

// Constituent getters and setters
const PFJetRef& PFTau::jetRef() const { return jetRef_; }
void PFTau::setjetRef(const PFJetRef& x) { jetRef_ = x; }

const PFTauTagInfoRef& PFTau::pfTauTagInfoRef() const {
//  edm::LogWarning("DeprecatedPFTauMember")
//      << "The PFTauTagInfoRef member is deprecated in the PFTau."
//      << " For access to the underlying PFJet, please use the jetRef() method";
  return PFTauTagInfoRef_;
}

void PFTau::setpfTauTagInfoRef(const PFTauTagInfoRef x) { PFTauTagInfoRef_ = x; }

const PFCandidatePtr& PFTau::leadPFChargedHadrCand() const { return leadPFChargedHadrCand_; }
const PFCandidatePtr& PFTau::leadPFNeutralCand() const { return leadPFNeutralCand_; }
const PFCandidatePtr& PFTau::leadPFCand() const { return leadPFCand_; }

void PFTau::setleadPFChargedHadrCand(const PFCandidatePtr& myLead) { leadPFChargedHadrCand_ = myLead;}
void PFTau::setleadPFNeutralCand(const PFCandidatePtr& myLead) { leadPFNeutralCand_ = myLead;}
void PFTau::setleadPFCand(const PFCandidatePtr& myLead) { leadPFCand_ = myLead;}

float PFTau::leadPFChargedHadrCandsignedSipt() const { return leadPFChargedHadrCandsignedSipt_; }
void PFTau::setleadPFChargedHadrCandsignedSipt(const float& x){ leadPFChargedHadrCandsignedSipt_ = x; }

const std::vector<PFCandidatePtr>& PFTau::signalPFCands() const { return selectedSignalPFCands_; }
void PFTau::setsignalPFCands(const std::vector<PFCandidatePtr>& myParts)  { selectedSignalPFCands_ = myParts; }
const std::vector<PFCandidatePtr>& PFTau::signalPFChargedHadrCands() const { return selectedSignalPFChargedHadrCands_; }
void PFTau::setsignalPFChargedHadrCands(const std::vector<PFCandidatePtr>& myParts)  { selectedSignalPFChargedHadrCands_ = myParts; }
const std::vector<PFCandidatePtr>& PFTau::signalPFNeutrHadrCands() const {return selectedSignalPFNeutrHadrCands_; }
void PFTau::setsignalPFNeutrHadrCands(const std::vector<PFCandidatePtr>& myParts)  { selectedSignalPFNeutrHadrCands_ = myParts; }
const std::vector<PFCandidatePtr>& PFTau::signalPFGammaCands() const { return selectedSignalPFGammaCands_; }
void PFTau::setsignalPFGammaCands(const std::vector<PFCandidatePtr>& myParts)  { selectedSignalPFGammaCands_ = myParts; }

const std::vector<PFCandidatePtr>& PFTau::isolationPFCands() const { return selectedIsolationPFCands_; }
void PFTau::setisolationPFCands(const std::vector<PFCandidatePtr>& myParts)  { selectedIsolationPFCands_ = myParts;} 
const std::vector<PFCandidatePtr>& PFTau::isolationPFChargedHadrCands() const { return selectedIsolationPFChargedHadrCands_; }
void PFTau::setisolationPFChargedHadrCands(const std::vector<PFCandidatePtr>& myParts)  { selectedIsolationPFChargedHadrCands_ = myParts; }
const std::vector<PFCandidatePtr>& PFTau::isolationPFNeutrHadrCands() const { return selectedIsolationPFNeutrHadrCands_; }
void PFTau::setisolationPFNeutrHadrCands(const std::vector<PFCandidatePtr>& myParts)  { selectedIsolationPFNeutrHadrCands_ = myParts; }
const std::vector<PFCandidatePtr>& PFTau::isolationPFGammaCands() const { return selectedIsolationPFGammaCands_; }
void PFTau::setisolationPFGammaCands(const std::vector<PFCandidatePtr>& myParts)  { selectedIsolationPFGammaCands_ = myParts; }

// PiZero and decay mode information
const std::vector<RecoTauPiZero>& PFTau::signalPiZeroCandidates() const {
  // Check if the signal pi zeros are already filled
  if ( signalPiZeroCandidates_.size() < signalPiZeroCandidatesRefs_.size() ) {
    // Fill them from the refs
    for ( size_t i = 0; i < signalPiZeroCandidatesRefs_.size(); ++i ) {
      signalPiZeroCandidates_.push_back(*signalPiZeroCandidatesRefs_[i]);
    }
  }
  return signalPiZeroCandidates_;
}

void PFTau::setsignalPiZeroCandidates(const std::vector<RecoTauPiZero>& cands) {
   signalPiZeroCandidates_ = cands;
}

void PFTau::setSignalPiZeroCandidatesRefs(const RecoTauPiZeroRefVector& cands) {
   signalPiZeroCandidatesRefs_ = cands;
}

void PFTau::setIsolationPiZeroCandidatesRefs(const RecoTauPiZeroRefVector& cands) {
   isolationPiZeroCandidatesRefs_ = cands;
}

const std::vector<RecoTauPiZero>& PFTau::isolationPiZeroCandidates() const {
  // Check if the signal pi zeros are already filled
  if ( isolationPiZeroCandidates_.size() < isolationPiZeroCandidatesRefs_.size() ) {
    // Fill them from the refs
    for ( size_t i = 0; i < isolationPiZeroCandidatesRefs_.size(); ++i ) {
      isolationPiZeroCandidates_.push_back(*isolationPiZeroCandidatesRefs_[i]);
    }
  }
  return isolationPiZeroCandidates_;
}

void PFTau::setisolationPiZeroCandidates(const std::vector<RecoTauPiZero>& cands) {
   signalPiZeroCandidates_ = cands;
}

// Tau Charged Hadron information
PFRecoTauChargedHadronRef PFTau::leadTauChargedHadronCandidate() const {
  if ( signalTauChargedHadronCandidatesRefs_.size() > 0 ) {
    return signalTauChargedHadronCandidatesRefs_[0];
  } else {
    return PFRecoTauChargedHadronRef();
  }
}

const std::vector<PFRecoTauChargedHadron>& PFTau::signalTauChargedHadronCandidates() const {
  // Check if the signal tau charged hadrons are already filled
  if ( signalTauChargedHadronCandidates_.size() < signalTauChargedHadronCandidatesRefs_.size() ) {
    // Fill them from the refs
    for ( size_t i = 0; i < signalTauChargedHadronCandidatesRefs_.size(); ++i ) {
      signalTauChargedHadronCandidates_.push_back(*signalTauChargedHadronCandidatesRefs_[i]);
    }
  }
  return signalTauChargedHadronCandidates_;
}

void PFTau::setSignalTauChargedHadronCandidates(const std::vector<PFRecoTauChargedHadron>& cands) {
  signalTauChargedHadronCandidates_ = cands;
}

void PFTau::setSignalTauChargedHadronCandidatesRefs(const PFRecoTauChargedHadronRefVector& cands) {
  signalTauChargedHadronCandidatesRefs_ = cands;
}

const std::vector<PFRecoTauChargedHadron>& PFTau::isolationTauChargedHadronCandidates() const {
  // Check if the isolation tau charged hadrons are already filled
  if ( isolationTauChargedHadronCandidates_.size() < isolationTauChargedHadronCandidatesRefs_.size() ) {
    // Fill them from the refs
    for ( size_t i = 0; i < isolationTauChargedHadronCandidatesRefs_.size(); ++i ) {
      isolationTauChargedHadronCandidates_.push_back(*isolationTauChargedHadronCandidatesRefs_[i]);
    }
  }
  return isolationTauChargedHadronCandidates_;
}

void PFTau::setIsolationTauChargedHadronCandidates(const std::vector<PFRecoTauChargedHadron>& cands) {
  isolationTauChargedHadronCandidates_ = cands;
}

void PFTau::setIsolationTauChargedHadronCandidatesRefs(const PFRecoTauChargedHadronRefVector& cands) {
  isolationTauChargedHadronCandidatesRefs_ = cands;
}

PFTau::hadronicDecayMode PFTau::decayMode() const {
  unsigned int nCharged = signalTauChargedHadronCandidates().size();
  unsigned int nPiZeros = signalPiZeroCandidates().size();
  // If no tracks exist, this is definitely not a tau!
  if ( !nCharged ) return kNull;
  // Find the maximum number of PiZeros our parameterization can hold
  const unsigned int maxPiZeros = kOneProngNPiZero;
  // Determine our track index
  unsigned int trackIndex = (nCharged - 1)*(maxPiZeros + 1);
  // Check if we handle the given number of tracks
  if ( trackIndex >= kRareDecayMode ) return kRareDecayMode;
  
  nPiZeros = ( nPiZeros <= maxPiZeros ) ? nPiZeros : maxPiZeros;
  return static_cast<PFTau::hadronicDecayMode>(trackIndex + nPiZeros);
}

// Setting information about the isolation region
float PFTau::isolationPFChargedHadrCandsPtSum() const {return isolationPFChargedHadrCandsPtSum_;}
void PFTau::setisolationPFChargedHadrCandsPtSum(const float& x){isolationPFChargedHadrCandsPtSum_=x;}

float PFTau::isolationPFGammaCandsEtSum() const {return isolationPFGammaCandsEtSum_;}
void PFTau::setisolationPFGammaCandsEtSum(const float& x){isolationPFGammaCandsEtSum_=x;}

float PFTau::maximumHCALPFClusterEt() const {return maximumHCALPFClusterEt_;}
void PFTau::setmaximumHCALPFClusterEt(const float& x){maximumHCALPFClusterEt_=x;}

// Electron variables
float PFTau::emFraction() const {return emFraction_;}
float PFTau::hcalTotOverPLead() const {return hcalTotOverPLead_;}
float PFTau::hcalMaxOverPLead() const {return hcalMaxOverPLead_;}
float PFTau::hcal3x3OverPLead() const {return hcal3x3OverPLead_;}
float PFTau::ecalStripSumEOverPLead() const {return ecalStripSumEOverPLead_;}
float PFTau::bremsRecoveryEOverPLead() const {return bremsRecoveryEOverPLead_;}
reco::TrackRef PFTau::electronPreIDTrack() const {return electronPreIDTrack_;}
float PFTau::electronPreIDOutput() const {return electronPreIDOutput_;}
bool PFTau::electronPreIDDecision() const {return electronPreIDDecision_;}

void PFTau::setemFraction(const float& x) {emFraction_ = x;}
void PFTau::sethcalTotOverPLead(const float& x) {hcalTotOverPLead_ = x;}
void PFTau::sethcalMaxOverPLead(const float& x) {hcalMaxOverPLead_ = x;}
void PFTau::sethcal3x3OverPLead(const float& x) {hcal3x3OverPLead_ = x;}
void PFTau::setecalStripSumEOverPLead(const float& x) {ecalStripSumEOverPLead_ = x;}
void PFTau::setbremsRecoveryEOverPLead(const float& x) {bremsRecoveryEOverPLead_ = x;}
void PFTau::setelectronPreIDTrack(const reco::TrackRef& x) {electronPreIDTrack_ = x;}
void PFTau::setelectronPreIDOutput(const float& x) {electronPreIDOutput_ = x;}
void PFTau::setelectronPreIDDecision(const bool& x) {electronPreIDDecision_ = x;}

// Muon variables
bool PFTau::hasMuonReference() const { // check if muon ref exists
    if( leadPFChargedHadrCand_.isNull() ) return false;
    else if( leadPFChargedHadrCand_.isNonnull() ){
        reco::MuonRef muonRef = leadPFChargedHadrCand_->muonRef();
        if( muonRef.isNull() )   return false;
        else if( muonRef.isNonnull() ) return  true;
    }
    return false;
}

float PFTau::caloComp() const {return caloComp_;}
float PFTau::segComp() const {return segComp_;}
bool  PFTau::muonDecision() const {return muonDecision_;}
void PFTau::setCaloComp(const float& x) {caloComp_ = x;}
void PFTau::setSegComp (const float& x) {segComp_  = x;}
void PFTau::setMuonDecision(const bool& x) {muonDecision_ = x;}

CandidatePtr PFTau::sourceCandidatePtr( size_type i ) const {
    if ( i!=0 ) return CandidatePtr();
    return refToPtr(jetRef());
}


bool PFTau::overlap(const Candidate& theCand) const {
    const RecoCandidate* theRecoCand = dynamic_cast<const RecoCandidate *>(&theCand);
    return (theRecoCand!=0 && (checkOverlap(track(), theRecoCand->track())));
}

void PFTau::dump(std::ostream& out) const {

    if(!out) return;

    if (pfTauTagInfoRef().isNonnull()) {
      out << "Its TauTagInfo constituents :"<<std::endl;
      out<<"# Tracks "<<pfTauTagInfoRef()->Tracks().size()<<std::endl;
      out<<"# PF charged hadr. cand's "<<pfTauTagInfoRef()->PFChargedHadrCands().size()<<std::endl;
      out<<"# PF neutral hadr. cand's "<<pfTauTagInfoRef()->PFNeutrHadrCands().size()<<std::endl;
      out<<"# PF gamma cand's "<<pfTauTagInfoRef()->PFGammaCands().size()<<std::endl;
    }
    if (jetRef().isNonnull()) {
      out << "Its constituents :"<< std::endl;
      out<<"# PF charged hadr. cand's "<< jetRef()->chargedHadronMultiplicity()<<std::endl;
      out<<"# PF neutral hadr. cand's "<< jetRef()->neutralHadronMultiplicity()<<std::endl;
      out<<"# PF gamma cand's "<< jetRef()->photonMultiplicity()<<std::endl;
      out<<"# Electron cand's "<< jetRef()->electronMultiplicity()<<std::endl;
    }
    out<<"in detail :"<<std::endl;

    out<<"Pt of the PFTau "<<pt()<<std::endl;
    const PFCandidatePtr& theLeadPFCand = leadPFChargedHadrCand();
    if(!theLeadPFCand){
        out<<"No Lead PFCand "<<std::endl;
    }else{
        out<<"Lead PFCand Particle Id " << (*theLeadPFCand).particleId() << std::endl;
        out<<"Lead PFCand Pt "<<(*theLeadPFCand).pt()<<std::endl;
        out<<"Lead PFCand Charge "<<(*theLeadPFCand).charge()<<std::endl;
        out<<"Lead PFCand TrkRef "<<(*theLeadPFCand).trackRef().isNonnull()<<std::endl;
        out<<"Inner point position (x,y,z) of the PFTau ("<<vx()<<","<<vy()<<","<<vz()<<")"<<std::endl;
        out<<"Charge of the PFTau "<<charge()<<std::endl;
        out<<"Et of the highest Et HCAL PFCluster "<<maximumHCALPFClusterEt()<<std::endl;
        out<<"Number of SignalPFChargedHadrCands = "<<signalPFChargedHadrCands().size()<<std::endl;
        out<<"Number of SignalPFGammaCands = "<<signalPFGammaCands().size()<<std::endl;
        out<<"Number of IsolationPFChargedHadrCands = "<<isolationPFChargedHadrCands().size()<<std::endl;
        out<<"Number of IsolationPFGammaCands = "<<isolationPFGammaCands().size()<<std::endl;
        out<<"Sum of Pt of charged hadr. PFCandidates in isolation annulus around Lead PF = "<<isolationPFChargedHadrCandsPtSum()<<std::endl;
        out<<"Sum of Et of gamma PFCandidates in other isolation annulus around Lead PF = "<<isolationPFGammaCandsEtSum()<<std::endl;

    }
    // return out;
}

std::ostream& operator<<(std::ostream& out, const reco::PFTau& tau) {

   if(!out) return out;

   out << std::setprecision(3)
     <<"PFTau "
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
