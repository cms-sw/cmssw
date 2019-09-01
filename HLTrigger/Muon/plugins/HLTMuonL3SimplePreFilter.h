#ifndef HLTMuonL3SimplePreFilter_h
#define HLTMuonL3SimplePreFilter_h

/** \class HLTMuonL3SimplePreFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing
 *  the a simple filtering for HLT muons 
 * 
 *  Original author:  S. Folgueras <santiago.folgueras@cern.ch>
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class HLTMuonL3SimplePreFilter : public HLTFilter {
public:
  explicit HLTMuonL3SimplePreFilter(const edm::ParameterSet &);
  ~HLTMuonL3SimplePreFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  bool hltFilter(edm::Event &,
                 const edm::EventSetup &,
                 trigger::TriggerFilterObjectWithRefs &filterproduct) const override;

private:
  static bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &,
                                      const std::vector<reco::RecoChargedCandidateRef> &);

  edm::InputTag candTag_;                                             // input tag identifying muon container
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_;  // token identifying muon container
  edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      previousCandToken_;  // token identifying product contains muons passing the previous level
  edm::InputTag beamspotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;

  const int min_N_;                  // minimum number of muons to fire the trigger
  const double max_Eta_;             // Eta cut
  const int min_Nhits_;              // threshold on number of hits on muon
  const double max_Dz_;              // dz cut
  const double min_DxySig_;          // dxy significance cut
  const double min_Pt_;              // pt threshold in GeV
  const double nsigma_Pt_;           // pt uncertainty margin (in number of sigmas)
  const double max_NormalizedChi2_;  // cutoff in normalized chi2
  const double max_DXYBeamSpot_;     // cutoff in dxy from the beamspot
  const double min_DXYBeamSpot_;     // minimum cut on dxy from the beamspot
  const int min_NmuonHits_;          // cutoff in minumum number of chi2 hits
  const double max_PtDifference_;    // cutoff in maximum different between global track and tracker track
  const double min_TrackPt_;         // cutoff in tracker track pt
  bool matchPreviousCand_;
};
#endif  //HLTMuonL3SimplePreFilter_h
