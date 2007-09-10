#include "DataFormats/TauReco/interface/CaloTau.h"

CaloTau::CaloTau() {  
  highestEtHCALhitEt_ = NAN;
  leadTracksignedSipt_=NAN;
  signalTracksInvariantMass_ = NAN; 
  TracksInvariantMass_ = NAN;
  isolationTracksPtSum_=NAN;
  isolationECALhitsEtSum_=NAN;
}

CaloTau::CaloTau(Charge q,const LorentzVector& p4,const Point& vtx) : BaseTau(q,p4,vtx) {
  highestEtHCALhitEt_ = NAN;
  leadTracksignedSipt_ =NAN;
  signalTracksInvariantMass_ = NAN; 
  TracksInvariantMass_ = NAN;
  isolationTracksPtSum_=NAN;
  isolationECALhitsEtSum_=NAN;
}

CaloTau* CaloTau::clone()const{return new CaloTau(*this);}

const CaloTauTagInfoRef& CaloTau::caloTauTagInfoRef()const{return CaloTauTagInfoRef_;}
void CaloTau::setcaloTauTagInfoRef(const CaloTauTagInfoRef x) {CaloTauTagInfoRef_=x;}

float CaloTau::leadTracksignedSipt()const{return leadTracksignedSipt_;}
void CaloTau::setleadTracksignedSipt(const float& x){leadTracksignedSipt_=x;}

float CaloTau::signalTracksInvariantMass()const{return signalTracksInvariantMass_;}
void CaloTau::setsignalTracksInvariantMass(const float& x){signalTracksInvariantMass_=x;}

float CaloTau::TracksInvariantMass()const{return TracksInvariantMass_;}
void CaloTau::setTracksInvariantMass(const float& x){TracksInvariantMass_=x;}

float CaloTau::isolationTracksPtSum()const{return isolationTracksPtSum_;}
void CaloTau::setisolationTracksPtSum(const float& x){isolationTracksPtSum_=x;}

float CaloTau::isolationECALhitsEtSum()const{return isolationECALhitsEtSum_;}
void CaloTau::setisolationECALhitsEtSum(const float& x){isolationECALhitsEtSum_=x;}

float CaloTau::highestEtHCALhitEt()const{return highestEtHCALhitEt_;}
void CaloTau::sethighestEtHCALhitEt(const float& x){highestEtHCALhitEt_=x;}

bool CaloTau::overlap(const Candidate& theCand)const{
  const RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
  return (theRecoCand!=0 && (checkOverlap(track(),theRecoCand->track())));
}
