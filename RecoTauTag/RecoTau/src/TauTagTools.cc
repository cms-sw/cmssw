#include "RecoTauTag/RecoTau/interface/TauTagTools.h"

using namespace reco;

namespace tautagtools {

  TrackRefVector filteredTracks(const TrackRefVector& theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2, const Vertex& pv){
    TrackRefVector filteredTracks;
    for(TrackRefVector::const_iterator iTk=theInitialTracks.begin();iTk!=theInitialTracks.end();iTk++){
      if ((**iTk).pt()>=tkminPt &&
	  (**iTk).normalizedChi2()<=tkmaxChi2 &&
	  fabs((**iTk).dxy(pv.position()))<=tkmaxipt &&
	  (**iTk).numberOfValidHits()>=tkminTrackerHitsn &&
	  (**iTk).hitPattern().numberOfValidPixelHits()>=tkminPixelHitsn)
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }
  TrackRefVector filteredTracks(const TrackRefVector& theInitialTracks, double tkminPt, int tkminPixelHitsn, int tkminTrackerHitsn, double tkmaxipt, double tkmaxChi2, double tktorefpointmaxDZ, const Vertex& pv, double refpoint_Z){
    TrackRefVector filteredTracks;
    for(TrackRefVector::const_iterator iTk=theInitialTracks.begin();iTk!=theInitialTracks.end();iTk++){
      if(pv.isFake()) tktorefpointmaxDZ=30.;
      if ((**iTk).pt()>=tkminPt &&
	  (**iTk).normalizedChi2()<=tkmaxChi2 &&
	  fabs((**iTk).dxy(pv.position()))<=tkmaxipt &&
	  (**iTk).numberOfValidHits()>=tkminTrackerHitsn &&
	  (**iTk).hitPattern().numberOfValidPixelHits()>=tkminPixelHitsn &&
	  fabs((**iTk).dz(pv.position()))<=tktorefpointmaxDZ)
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }

  std::vector<reco::CandidatePtr> filteredPFChargedHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double ChargedHadrCand_tkminPt, int ChargedHadrCand_tkminPixelHitsn, int ChargedHadrCand_tkminTrackerHitsn, double ChargedHadrCand_tkmaxipt, double ChargedHadrCand_tkmaxChi2, const Vertex& pv){
    std::vector<reco::CandidatePtr> filteredPFChargedHadrCands;
    for(std::vector<reco::CandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (std::abs((*iPFCand)->pdgId()) == 211 || std::abs((*iPFCand)->pdgId()) == 13 || std::abs((*iPFCand)->pdgId()) == 11){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	const reco::Track* PFChargedHadrCand_rectk = (*iPFCand)->bestTrack();
	if (PFChargedHadrCand_rectk != nullptr) {
	  if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	      (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	      fabs((*PFChargedHadrCand_rectk).dxy(pv.position()))<=ChargedHadrCand_tkmaxipt &&
	      (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn &&
	      (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn) 
	    filteredPFChargedHadrCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFChargedHadrCands;
  }
  std::vector<reco::CandidatePtr> filteredPFChargedHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands, double ChargedHadrCand_tkminPt, int ChargedHadrCand_tkminPixelHitsn, int ChargedHadrCand_tkminTrackerHitsn, double ChargedHadrCand_tkmaxipt, double ChargedHadrCand_tkmaxChi2, double ChargedHadrCand_tktorefpointmaxDZ, const Vertex& pv, double refpoint_Z){
    if(pv.isFake()) ChargedHadrCand_tktorefpointmaxDZ = 30.;
    std::vector<reco::CandidatePtr> filteredPFChargedHadrCands;
    for(std::vector<reco::CandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (std::abs((*iPFCand)->pdgId()) == 211 || std::abs((*iPFCand)->pdgId()) == 13 || std::abs((*iPFCand)->pdgId()) == 11){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	const reco::Track* PFChargedHadrCand_rectk = (*iPFCand)->bestTrack();
	if (PFChargedHadrCand_rectk != nullptr) {
	  if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	      (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	      fabs((*PFChargedHadrCand_rectk).dxy(pv.position()))<=ChargedHadrCand_tkmaxipt &&
	      (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn &&
	      (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn &&
	      fabs((*PFChargedHadrCand_rectk).dz(pv.position()))<=ChargedHadrCand_tktorefpointmaxDZ)
	    filteredPFChargedHadrCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFChargedHadrCands;
  }
  
  std::vector<reco::CandidatePtr> filteredPFNeutrHadrCands(const std::vector<reco::CandidatePtr>& theInitialPFCands,double NeutrHadrCand_HcalclusMinEt){
    std::vector<reco::CandidatePtr> filteredPFNeutrHadrCands;
    for(std::vector<reco::CandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (std::abs((*iPFCand)->pdgId()) == 130){
	// *** Whether the neutral hadron candidate will be selected or not depends on its rec. HCAL cluster properties. 
	if ((**iPFCand).et()>=NeutrHadrCand_HcalclusMinEt){
	  filteredPFNeutrHadrCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFNeutrHadrCands;
  }
  
  std::vector<reco::CandidatePtr> filteredPFGammaCands(const std::vector<reco::CandidatePtr>& theInitialPFCands,double GammaCand_EcalclusMinEt){
    std::vector<reco::CandidatePtr> filteredPFGammaCands;
    for(std::vector<reco::CandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (std::abs((*iPFCand)->pdgId()) == 22){
	// *** Whether the gamma candidate will be selected or not depends on its rec. ECAL cluster properties. 
	if ((**iPFCand).et()>=GammaCand_EcalclusMinEt){
	  filteredPFGammaCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFGammaCands;
  }

}
