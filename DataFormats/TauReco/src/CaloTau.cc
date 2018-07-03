#include "DataFormats/TauReco/interface/CaloTau.h"

using namespace reco;

CaloTau::CaloTau() {  
  maximumHCALhitEt_ = NAN;
  leadTracksignedSipt_=NAN;
  leadTrackHCAL3x3hitsEtSum_=NAN;
  leadTrackHCAL3x3hottesthitDEta_=NAN;
  signalTracksInvariantMass_ = NAN; 
  TracksInvariantMass_ = NAN;
  isolationTracksPtSum_=NAN;
  isolationECALhitsEtSum_=NAN;
}

CaloTau::CaloTau(Charge q,const LorentzVector& p4,const Point& vtx) : BaseTau(q,p4,vtx) {
  maximumHCALhitEt_ = NAN;
  leadTracksignedSipt_ =NAN;
  leadTrackHCAL3x3hitsEtSum_=NAN;
  leadTrackHCAL3x3hottesthitDEta_=NAN;
  signalTracksInvariantMass_ = NAN; 
  TracksInvariantMass_ = NAN;
  isolationTracksPtSum_=NAN;
  isolationECALhitsEtSum_=NAN;
}

CaloTau* CaloTau::clone()const{return new CaloTau(*this);}

const CaloTauTagInfoRef& CaloTau::caloTauTagInfoRef()const{return CaloTauTagInfoRef_;}
void CaloTau::setcaloTauTagInfoRef(const CaloTauTagInfoRef x) {CaloTauTagInfoRef_=x;}

const CaloJetRef CaloTau::rawJetRef() const {
	return this->caloTauTagInfoRef()->calojetRef();
}

float CaloTau::leadTracksignedSipt()const{return leadTracksignedSipt_;}
void CaloTau::setleadTracksignedSipt(const float& x){leadTracksignedSipt_=x;}

float CaloTau::leadTrackHCAL3x3hitsEtSum()const{return leadTrackHCAL3x3hitsEtSum_;}
void CaloTau::setleadTrackHCAL3x3hitsEtSum(const float& x){leadTrackHCAL3x3hitsEtSum_=x;}

float CaloTau::leadTrackHCAL3x3hottesthitDEta()const{return leadTrackHCAL3x3hottesthitDEta_;}
void CaloTau::setleadTrackHCAL3x3hottesthitDEta(const float& x){leadTrackHCAL3x3hottesthitDEta_=x;}

float CaloTau::signalTracksInvariantMass()const{return signalTracksInvariantMass_;}
void CaloTau::setsignalTracksInvariantMass(const float& x){signalTracksInvariantMass_=x;}

float CaloTau::TracksInvariantMass()const{return TracksInvariantMass_;}
void CaloTau::setTracksInvariantMass(const float& x){TracksInvariantMass_=x;}

float CaloTau::isolationTracksPtSum()const{return isolationTracksPtSum_;}
void CaloTau::setisolationTracksPtSum(const float& x){isolationTracksPtSum_=x;}

float CaloTau::isolationECALhitsEtSum()const{return isolationECALhitsEtSum_;}
void CaloTau::setisolationECALhitsEtSum(const float& x){isolationECALhitsEtSum_=x;}

float CaloTau::maximumHCALhitEt()const{return maximumHCALhitEt_;}
void CaloTau::setmaximumHCALhitEt(const float& x){maximumHCALhitEt_=x;}

bool CaloTau::overlap(const reco::Candidate& theCand)const{
  const reco::RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
  return (theRecoCand!=nullptr && (checkOverlap(track(),theRecoCand->track())));
}
