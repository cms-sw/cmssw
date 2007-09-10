#include "DataFormats/TauReco/interface/PFTau.h"

PFTau::PFTau(){
  PFCandidateRef pfLead;
  leadPFChargedHadrCand_ = pfLead;
  leadPFChargedHadrCandsignedSipt_=NAN;
  
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrCands_ =pfTmp;
  selectedSignalPFNeutrHadrCands_=pfTmp;
  selectedSignalPFGammaCands_ = pfTmp;

  selectedIsolationPFChargedHadrCands_ = pfTmp;
  selectedIsolationPFNeutrHadrCands_ = pfTmp;
  selectedIsolationPFGammaCands_ = pfTmp;
}

PFTau::PFTau(Charge q,const LorentzVector& p4,const Point& vtx) : BaseTau(q,p4,vtx){
  PFCandidateRef pfLead;
  leadPFChargedHadrCand_ = pfLead;
  leadPFChargedHadrCandsignedSipt_=NAN;
  
  PFCandidateRefVector pfTmp;
  selectedSignalPFChargedHadrCands_ =pfTmp;
  selectedSignalPFNeutrHadrCands_=pfTmp;
  selectedSignalPFGammaCands_ = pfTmp;

  selectedIsolationPFChargedHadrCands_ = pfTmp;
  selectedIsolationPFNeutrHadrCands_ = pfTmp;
  selectedIsolationPFGammaCands_ = pfTmp;
}

PFTau* PFTau::clone()const{return new PFTau(*this);}

const PFTauTagInfoRef& PFTau::pfTauTagInfoRef()const{return PFTauTagInfoRef_;}
void PFTau::setpfTauTagInfoRef(const PFTauTagInfoRef x) {PFTauTagInfoRef_=x;}
    
const PFCandidateRef& PFTau::leadPFChargedHadrCand()const {return leadPFChargedHadrCand_;}   
void PFTau::setleadPFChargedHadrCand(const PFCandidateRef& myLead) { leadPFChargedHadrCand_=myLead;}   
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

float PFTau::highestEtHCALPFClusterEt()const{return highestEtHCALPFClusterEt_;}
void PFTau::sethighestEtHCALPFClusterEt(const float& x){highestEtHCALPFClusterEt_=x;}

bool PFTau::overlap(const Candidate& theCand)const{
  const RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
  return (theRecoCand!=0 && (checkOverlap(track(),theRecoCand->track())));
}
