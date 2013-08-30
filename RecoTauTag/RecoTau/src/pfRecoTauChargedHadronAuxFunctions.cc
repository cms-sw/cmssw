#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include <TMath.h>

namespace reco { namespace tau {

void setChargedHadronP4(reco::PFRecoTauChargedHadron& chargedHadron, double scaleFactor_neutralPFCands)
{
  double chargedHadronP     = 0.;
  double chargedHadronTheta = 0.;
  double chargedHadronPhi   = 0.;
  if ( chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate) ||
       chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kPFNeutralHadron)   ) {
    const reco::PFCandidatePtr& chargedPFCand = chargedHadron.getChargedPFCandidate();
    assert(chargedPFCand.isNonnull());
    chargedHadronP     += chargedPFCand->p();
    chargedHadronTheta  = chargedPFCand->theta();
    chargedHadronPhi    = chargedPFCand->phi();
  } else if ( chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kTrack) ) {
    const reco::PFRecoTauChargedHadron::TrackPtr& track = chargedHadron.getTrack();
    assert(track.isNonnull());
    chargedHadronP     += track->p();
    chargedHadronTheta  = track->theta();
    chargedHadronPhi    = track->phi();
  } else assert(0);
  const std::vector<reco::PFCandidatePtr>& neutralPFCands = chargedHadron.getNeutralPFCandidates();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator neutralPFCand = neutralPFCands.begin();
	neutralPFCand != neutralPFCands.end(); ++neutralPFCand ) {
    chargedHadronP += scaleFactor_neutralPFCands*(*neutralPFCand)->p();
  }
      
  reco::Candidate::LorentzVector chargedHadronP4 = compChargedHadronP4(chargedHadronP, chargedHadronTheta, chargedHadronPhi);
  chargedHadron.setP4(chargedHadronP4);
}

reco::Candidate::LorentzVector compChargedHadronP4(double chargedHadronP, double chargedHadronTheta, double chargedHadronPhi)
{
  const double chargedPionMass = 0.13957; // GeV
  double chargedHadronEn = sqrt(chargedHadronP*chargedHadronP + chargedPionMass*chargedPionMass);  
  double chargedHadronPx = chargedHadronP*TMath::Cos(chargedHadronPhi)*TMath::Sin(chargedHadronTheta);
  double chargedHadronPy = chargedHadronP*TMath::Sin(chargedHadronPhi)*TMath::Sin(chargedHadronTheta);
  double chargedHadronPz = chargedHadronP*TMath::Cos(chargedHadronTheta);    
  reco::Candidate::LorentzVector chargedHadronP4(chargedHadronPx, chargedHadronPy, chargedHadronPz, chargedHadronEn);
  return chargedHadronP4;
}

}} // end namespace reco::tau
