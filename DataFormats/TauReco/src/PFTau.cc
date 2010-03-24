#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Common/interface/RefToPtr.h"

using namespace reco;
using namespace std;


PFTau::PFTau()
{
    leadPFChargedHadrCandsignedSipt_=NAN;
    isolationPFChargedHadrCandsPtSum_=NAN;
    isolationPFGammaCandsEtSum_=NAN;
    maximumHCALPFClusterEt_=NAN;
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

PFTau::PFTau(Charge q,const LorentzVector& p4,const Point& vtx) : BaseTau(q,p4,vtx)
{
    leadPFChargedHadrCandsignedSipt_=NAN;    
    isolationPFChargedHadrCandsPtSum_=NAN;
    isolationPFGammaCandsEtSum_=NAN;
    maximumHCALPFClusterEt_=NAN;
    
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

PFTau* PFTau::clone()const{return new PFTau(*this);}

const PFTauTagInfoRef& PFTau::pfTauTagInfoRef()const{return PFTauTagInfoRef_;}
void PFTau::setpfTauTagInfoRef(const PFTauTagInfoRef x) {PFTauTagInfoRef_=x;}

const PFCandidateRef& PFTau::leadPFChargedHadrCand()const {return leadPFChargedHadrCand_;}   
const PFCandidateRef& PFTau::leadPFNeutralCand()const {return leadPFNeutralCand_;}   
const PFCandidateRef& PFTau::leadPFCand()const {return leadPFCand_;}   

void PFTau::setleadPFChargedHadrCand(const PFCandidateRef& myLead) { leadPFChargedHadrCand_=myLead;}   
void PFTau::setleadPFNeutralCand(const PFCandidateRef& myLead) { leadPFNeutralCand_=myLead;}   
void PFTau::setleadPFCand(const PFCandidateRef& myLead) { leadPFCand_=myLead;}   

float PFTau::leadPFChargedHadrCandsignedSipt()const{return leadPFChargedHadrCandsignedSipt_;}
void PFTau::setleadPFChargedHadrCandsignedSipt(const float& x){leadPFChargedHadrCandsignedSipt_=x;}

const PFCandidateRefVector& PFTau::signalPFCands()const {return selectedSignalPFCands_;}
void PFTau::setsignalPFCands(const PFCandidateRefVector& myParts)  { selectedSignalPFCands_ = myParts;}
const PFCandidateRefVector& PFTau::signalPFChargedHadrCands()const {return selectedSignalPFChargedHadrCands_;}
void PFTau::setsignalPFChargedHadrCands(const PFCandidateRefVector& myParts)  { selectedSignalPFChargedHadrCands_ = myParts;}
const PFCandidateRefVector& PFTau::signalPFNeutrHadrCands()const {return selectedSignalPFNeutrHadrCands_;}
void PFTau::setsignalPFNeutrHadrCands(const PFCandidateRefVector& myParts)  { selectedSignalPFNeutrHadrCands_ = myParts;}
const PFCandidateRefVector& PFTau::signalPFGammaCands()const {return selectedSignalPFGammaCands_;}
void PFTau::setsignalPFGammaCands(const PFCandidateRefVector& myParts)  { selectedSignalPFGammaCands_ = myParts;}

const PFCandidateRefVector& PFTau::isolationPFCands()const {return selectedIsolationPFCands_;}
void PFTau::setisolationPFCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFCands_ = myParts;}
const PFCandidateRefVector& PFTau::isolationPFChargedHadrCands()const {return selectedIsolationPFChargedHadrCands_;}
void PFTau::setisolationPFChargedHadrCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFChargedHadrCands_ = myParts;}
const PFCandidateRefVector& PFTau::isolationPFNeutrHadrCands()const {return selectedIsolationPFNeutrHadrCands_;}
void PFTau::setisolationPFNeutrHadrCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFNeutrHadrCands_ = myParts;}
const PFCandidateRefVector& PFTau::isolationPFGammaCands()const {return selectedIsolationPFGammaCands_;}
void PFTau::setisolationPFGammaCands(const PFCandidateRefVector& myParts)  { selectedIsolationPFGammaCands_ = myParts;}

float PFTau::isolationPFChargedHadrCandsPtSum()const{return isolationPFChargedHadrCandsPtSum_;}
void PFTau::setisolationPFChargedHadrCandsPtSum(const float& x){isolationPFChargedHadrCandsPtSum_=x;}

float PFTau::isolationPFGammaCandsEtSum()const{return isolationPFGammaCandsEtSum_;}
void PFTau::setisolationPFGammaCandsEtSum(const float& x){isolationPFGammaCandsEtSum_=x;}

float PFTau::maximumHCALPFClusterEt()const{return maximumHCALPFClusterEt_;}
void PFTau::setmaximumHCALPFClusterEt(const float& x){maximumHCALPFClusterEt_=x;}

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

bool PFTau::hasMuonReference()const{ // check if muon ref exists
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
//


CandidatePtr PFTau::sourceCandidatePtr( size_type i ) const {
    if( i!=0 ) return CandidatePtr();
    
    const PFJetRef& pfJetRef = pfTauTagInfoRef()->pfjetRef();
    return  refToPtr( pfJetRef );
}


bool PFTau::overlap(const Candidate& theCand)const{
    const RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
    return (theRecoCand!=0 && (checkOverlap(track(),theRecoCand->track())));
}

void PFTau::dump(std::ostream& out) const {
    
    if(!out) return;
    
    out << "Its constituents :"<<std::endl;
    out<<"# Tracks "<<pfTauTagInfoRef()->Tracks().size()<<std::endl;
    out<<"# PF charged hadr. cand's "<<pfTauTagInfoRef()->PFChargedHadrCands().size()<<std::endl;
    out<<"# PF neutral hadr. cand's "<<pfTauTagInfoRef()->PFNeutrHadrCands().size()<<std::endl;
    out<<"# PF gamma cand's "<<pfTauTagInfoRef()->PFGammaCands().size()<<std::endl;
    out<<"in detail :"<<std::endl;
    
    out<<"Pt of the PFTau "<<pt()<<std::endl;
    PFCandidateRef theLeadPFCand = leadPFChargedHadrCand();
    if(!theLeadPFCand){
        out<<"No Lead PFCand "<<std::endl;
    }else{
        out<<"Lead PFCand Pt "<<(*theLeadPFCand).pt()<<std::endl;
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

namespace reco {
    
    
    std::ostream& operator<<(std::ostream& out, const reco::PFTau& tau) {
        
        if(!out) return out;
        
        out<<"PFTau "
        <<tau.pt()<<","
        <<tau.eta()<<","
        <<tau.phi()<<"  "    
        <<tau.signalPFCands().size()<<","
        <<tau.signalPFChargedHadrCands().size()<<","
        <<tau.signalPFGammaCands().size()<<","
        <<tau.signalPFNeutrHadrCands().size()<<"  "
        <<tau.isolationPFCands().size()<<","
        <<tau.isolationPFChargedHadrCands().size()<<","
        <<tau.isolationPFGammaCands().size()<<","
        <<tau.isolationPFNeutrHadrCands().size();
        
        return out;
    }
    
}
