#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

float HSCParticle::p() const {
  if(hasMuonCombinedTrack()) return combinedTrack().p();
  if(hasMuonStaTrack())      return staTrack().p();
  if(hasMuonTrack())         return muonTrack().p();
  if(hasTrackerTrack())      return trackerTrack().p();
  return 0.;
}

float HSCParticle::pt() const {
  if(hasMuonCombinedTrack()) return combinedTrack().pt();
  if(hasMuonStaTrack())      return staTrack().pt();
  if(hasMuonTrack())         return muonTrack().pt();
  if(hasTrackerTrack())      return trackerTrack().pt();
  return 0.;
}

float HSCParticle::massDtError() const {
  double ib2 = dt.invBeta*dt.invBeta;
  return dt.invBetaErr*(ib2/sqrt(ib2-1)) ;
}

float HSCParticle::massTkError() const {
  double dedxError = 0.2*sqrt(10./tk.nDedxHits())*0.4/tk.invBeta2();
  return dedxError/(2.*tk.invBeta2()-1);
}

float HSCParticle::massAvgError() const {
  double ptMassError=(track_->ptError()/track_->pt());
  ptMassError*=ptMassError;
  double ptMassError2=(muon_->track()->ptError()/muon_->track()->pt());
  ptMassError2*=ptMassError2;
  double dtMassError=massDtError();
  dtMassError*= dtMassError;
  double tkMassError = massTkError();
  tkMassError*=tkMassError;
  return sqrt(ptMassError/4.+ptMassError2/4.+dtMassError/4.+tkMassError/4.);
}

}
