#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco { namespace tau {

void setChargedHadronP4(reco::PFRecoTauChargedHadron& chargedHadron, double scaleFactor_neutralPFCands)
{
  double chargedHadronPx = 0.;
  double chargedHadronPy = 0.;
  double chargedHadronPz = 0.;
  if ( chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate) ||
       chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kPFNeutralHadron)   ) {
    const reco::PFCandidatePtr& chargedPFCand = chargedHadron.getChargedPFCandidate();
    assert(chargedPFCand.isNonnull());
    chargedHadronPx += chargedPFCand->px();
    chargedHadronPy += chargedPFCand->py();
    chargedHadronPz += chargedPFCand->pz();
  } else if ( chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kTrack) ) {
    const reco::PFRecoTauChargedHadron::TrackPtr& track = chargedHadron.getTrack();
    assert(track.isNonnull());
    chargedHadronPx += track->px();
    chargedHadronPy += track->py();
    chargedHadronPz += track->pz();
  } else assert(0);
  const std::vector<reco::PFCandidatePtr>& neutralPFCands = chargedHadron.getNeutralPFCandidates();
  for ( std::vector<reco::PFCandidatePtr>::const_iterator neutralPFCand = neutralPFCands.begin();
	neutralPFCand != neutralPFCands.end(); ++neutralPFCand ) {
    chargedHadronPx += scaleFactor_neutralPFCands*(*neutralPFCand)->px();
    chargedHadronPy += scaleFactor_neutralPFCands*(*neutralPFCand)->py();
    chargedHadronPz += scaleFactor_neutralPFCands*(*neutralPFCand)->pz();
  }
      
  reco::Candidate::LorentzVector chargedHadronP4 = compChargedHadronP4(chargedHadronPx, chargedHadronPy, chargedHadronPz);
  chargedHadron.setP4(chargedHadronP4);
}

reco::Candidate::LorentzVector compChargedHadronP4(double chargedHadronPx, double chargedHadronPy, double chargedHadronPz)
{
  const double chargedPionMass = 0.13957; // GeV
  double chargedHadronEn = sqrt(chargedHadronPx*chargedHadronPx + chargedHadronPy*chargedHadronPy + chargedHadronPz*chargedHadronPz + chargedPionMass*chargedPionMass);  
  reco::Candidate::LorentzVector chargedHadronP4(chargedHadronPx, chargedHadronPy, chargedHadronPz, chargedHadronEn);
  return chargedHadronP4;
}

}} // end namespace reco::tau
