#ifndef RecoTauTag_RecoTau_TauTagTools_h
#define RecoTauTag_RecoTau_TauTagTools_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace tautagtools {
  reco::TrackRefVector filteredTracks(const reco::TrackRefVector& theInitialTracks, double tkminPt, int tkminPixelHitsn, int tkminTrackerHitsn, double tkmaxipt, double tkmaxChi2, const reco::Vertex& pV);
  reco::TrackRefVector filteredTracks(const reco::TrackRefVector& theInitialTracks, double tkminPt, int tkminPixelHitsn, int tkminTrackerHitsn, double tkmaxipt, double tkmaxChi2, double tktorefpointmaxDZ, const reco::Vertex& pV, double refpoint_Z);

  std::vector<reco::CandidatePtr> filteredPFChargedHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double ChargedHadrCand_tkminPt, int ChargedHadrCand_tkminPixelHitsn, int ChargedHadrCand_tkminTrackerHitsn, double ChargedHadrCand_tkmaxipt, double ChargedHadrCand_tkmaxChi2, const reco::Vertex& pV);
  std::vector<reco::CandidatePtr> filteredPFChargedHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double ChargedHadrCand_tkminPt, int ChargedHadrCand_tkminPixelHitsn, int ChargedHadrCand_tkminTrackerHitsn, double ChargedHadrCand_tkmaxipt, double ChargedHadrCand_tkmaxChi2, double ChargedHadrCand_tktorefpointmaxDZ, const reco::Vertex& pV, double refpoint_Z);
  std::vector<reco::CandidatePtr> filteredPFNeutrHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double NeutrHadrCand_HcalclusMinEt);
  std::vector<reco::CandidatePtr> filteredPFGammaCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double GammaCand_EcalclusMinEt);
}

#endif
