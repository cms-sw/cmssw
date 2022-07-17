#ifndef HLTMuonDimuonL3Filter_h
#define HLTMuonDimuonL3Filter_h

/** \class HLTMuonDimuonL3Filter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a muon pair
 *  filter for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonDimuonL3Filter : public HLTFilter {
public:
  explicit HLTMuonDimuonL3Filter(const edm::ParameterSet&);
  ~HLTMuonDimuonL3Filter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  static bool triggeredByLevel2(reco::TrackRef const& track, std::vector<reco::RecoChargedCandidateRef> const& vcands);
  bool applyMuonSelection(const reco::RecoChargedCandidateRef&, const reco::BeamSpot&) const;
  bool applyDiMuonSelection(const reco::RecoChargedCandidateRef&,
                            const reco::RecoChargedCandidateRef&,
                            const reco::BeamSpot&,
                            const edm::ESHandle<MagneticField>&) const;

  const PropagateToMuonSetup propSetup_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldRecordToken_;
  const edm::InputTag beamspotTag_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  const edm::InputTag candTag_;  // input tag identifying product contains muons
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_;  // token identifying product contains muons
  const edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      previousCandToken_;          // tokenidentifying product contains muons passing the previous level
  const edm::InputTag l1CandTag_;  // input tag identifying product contains muons passing the L1 level
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      l1CandToken_;                // token identifying product contains muons passing the L1 level
  const edm::InputTag recoMuTag_;  // input tag identifying reco muons
  const edm::EDGetTokenT<reco::MuonCollection> recoMuToken_;  // token identifying product contains reco muons
  bool previousCandIsL2_;
  const bool fast_Accept_;  // flag to save time: stop processing after identification of the first valid pair
  const int min_N_;         // minimum number of muons to fire the trigger
  const double max_Eta_;    // Eta cut
  const int min_Nhits_;     // threshold on number of hits on muon
  const double max_Dr_;     // impact parameter cut
  const double max_Dz_;     // dz cut
  const int chargeOpt_;     // Charge option (0:nothing; +1:same charge, -1:opposite charge)
  const std::vector<double> min_PtPair_;   // minimum Pt for the dimuon system
  const std::vector<double> max_PtPair_;   // miaximum Pt for the dimuon system
  const std::vector<double> min_PtMax_;    // minimum Pt for muon with max Pt in pair
  const std::vector<double> min_PtMin_;    // minimum Pt for muon with min Pt in pair
  const std::vector<double> max_PtMin_;    // maximum Pt for muon with min Pt in pair
  const std::vector<double> min_InvMass_;  // minimum invariant mass of pair
  const std::vector<double> max_InvMass_;  // maximum invariant mass of pair
  const bool applyMinDiMuonDeltaR2Cut_;    // apply cut on minimum Delta-R^2 distance between the muons
  const double min_DiMuonDeltaR2_;         // minimum Delta-R^2 distance between the muons
  const double min_Acop_;                  // minimum acoplanarity
  const double max_Acop_;                  // maximum acoplanarity
  const double min_PtBalance_;             // minimum Pt difference
  const double max_PtBalance_;             // maximum Pt difference
  const double nsigma_Pt_;                 // pt uncertainty margin (in number of sigmas)
  const double max_DCAMuMu_;               // DCA between the two muons
  const double max_YPair_;                 // |rapidity| of pair
  const bool cutCowboys_;                  ///< if true, reject muon-track pairs that bend towards each other
  const edm::InputTag theL3LinksLabel;     //Needed to iterL3
  const edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;  //Needed to iterL3
  const double L1MatchingdR_;
  const double L1MatchingdR2_;
  const bool matchPreviousCand_;
  const double MuMass2_;
};

#endif  //HLTMuonDimuonFilter_h
