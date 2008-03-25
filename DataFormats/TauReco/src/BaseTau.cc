#include "DataFormats/TauReco/interface/BaseTau.h"

BaseTau::BaseTau() {
  alternatLorentzVect_.SetPx(NAN);
  alternatLorentzVect_.SetPy(NAN);
  alternatLorentzVect_.SetPz(NAN);
  alternatLorentzVect_.SetE(NAN);
    
  TrackRefVector tmp;
  TrackRef leadTk;
  leadTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
}

BaseTau::BaseTau(Charge q,const LorentzVector& p4,const Point& vtx) : RecoCandidate(q,p4,vtx,-15*q){
  alternatLorentzVect_.SetPx(NAN);
  alternatLorentzVect_.SetPy(NAN);
  alternatLorentzVect_.SetPz(NAN);
  alternatLorentzVect_.SetE(NAN);
    
  TrackRefVector tmp;
  TrackRef leadTk;
  leadTrack_ = leadTk;
  signalTracks_ =tmp ;
  isolationTracks_= tmp;
}

BaseTau* BaseTau::clone()const{return new BaseTau(*this);}

math::XYZTLorentzVector BaseTau::alternatLorentzVect()const{return(alternatLorentzVect_);} 
void BaseTau::setalternatLorentzVect(math::XYZTLorentzVector x){alternatLorentzVect_=x;}
    
const TrackRef& BaseTau::leadTrack() const {return leadTrack_;}
void BaseTau::setleadTrack(const TrackRef& myTrack) { leadTrack_ = myTrack;}
const TrackRefVector& BaseTau::signalTracks() const {return signalTracks_;}
void BaseTau::setsignalTracks(const TrackRefVector& myTracks)  { signalTracks_ = myTracks;}
const TrackRefVector& BaseTau::isolationTracks() const {return isolationTracks_;}
void BaseTau::setisolationTracks(const TrackRefVector& myTracks)  { isolationTracks_ = myTracks;}

bool BaseTau::overlap(const Candidate& theCand)const{
  const RecoCandidate* theRecoCand=dynamic_cast<const RecoCandidate *>(&theCand);
  return (theRecoCand!=0 && (checkOverlap(track(),theRecoCand->track())));
}
