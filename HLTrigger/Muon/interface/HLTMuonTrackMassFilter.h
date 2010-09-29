#ifndef HLTMuonTrackMassFilter_h_
#define HLTMuonTrackMassFilter_h_
/** HLT filter by muon+track mass (taken from two RecoChargedCandidate collections). 
 *  Tracks are subject to quality and momentum cuts. The match with previous muon
 *  (and possibly track, if run after another mass filter) candidates is checked.
*/
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <vector>


class HLTMuonTrackMassFilter : public HLTFilter {
public:
  explicit HLTMuonTrackMassFilter(const edm::ParameterSet&);
  ~HLTMuonTrackMassFilter() {}

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  bool pairMatched (std::vector<reco::RecoChargedCandidateRef>& prevMuonRefs,
		    std::vector<reco::RecoChargedCandidateRef>& prevTrackRefs,
		    const reco::RecoChargedCandidateRef& muonRef,
		    const reco::RecoChargedCandidateRef& trackRef) const;
      
private:
  edm::InputTag beamspotTag_;   ///< beamspot used for quality cuts
  edm::InputTag muonTag_;       ///< RecoChargedCandidateCollection (muons)
  edm::InputTag trackTag_;      ///< RecoChargedCandidateCollection (tracks)
  edm::InputTag prevCandTag_;   ///< filter objects from previous filter
  bool saveTag_;                ///< save tags in filter object collection?
  std::vector<double> minMasses_; ///< lower mass limits
  std::vector<double> maxMasses_; ///< higher mass limits
  bool checkCharge_;            ///< check opposite charge?
  double minTrackPt_;           ///< track pt cut
  double minTrackP_;            ///< track p cut
  double maxTrackEta_;          ///< track |eta| cut
  double maxTrackDxy_;          ///< track tip cut w.r.t. beamspot
  double maxTrackDz_;           ///< track lip cut w.r.t. beamspot
  int minTrackHits_;            ///< # valid hits on track
  double maxTrackNormChi2_;     ///< normalized chi2 of track
  double maxDzMuonTrack_;       ///< relative deltaZ between muon and track
  bool cutCowboys_;             ///< if true, reject muon-track pairs that bend towards each other
};

#endif
