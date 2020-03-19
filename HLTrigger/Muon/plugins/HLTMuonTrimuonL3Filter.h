#ifndef HLTMuonTrimuonL3Filter_h
#define HLTMuonTrimuonL3Filter_h

/** \class HLTMuonTrimuonL3Filter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a muon triplet
 *  filter for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonTrimuonL3Filter : public HLTFilter {
public:
  explicit HLTMuonTrimuonL3Filter(const edm::ParameterSet&);
  ~HLTMuonTrimuonL3Filter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  static bool triggeredByLevel2(const reco::TrackRef& track, std::vector<reco::RecoChargedCandidateRef>& vcands);

  edm::InputTag beamspotTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  edm::InputTag candTag_;                                             // input tag identifying product contains muons
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_;  // token identifying product contains muons
  edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      previousCandToken_;  // token identifying product contains muons passing the previous level

  bool fast_Accept_;      // flag to save time: stop processing after identification of the first valid triplet
  double max_Eta_;        // Eta cut
  int min_Nhits_;         // threshold on number of hits on muon
  double max_Dr_;         // impact parameter cut
  double max_Dz_;         // dz cut
  int chargeOpt_;         // Charge option (0:nothing; +1:same charge, -1:opposite charge)
  double min_PtTriplet_;  // minimum Pt for the dimuon system
  double min_PtMax_;      // minimum Pt for muon with max Pt in triplet
  double min_PtMin_;      // minimum Pt for muon with min Pt in triplet
  double min_InvMass_;    // minimum invariant mass of triplet
  double max_InvMass_;    // maximum invariant mass of triplet
  double min_Acop_;       // minimum acoplanarity
  double max_Acop_;       // maximum acoplanarity
  double min_PtBalance_;  // minimum Pt difference
  double max_PtBalance_;  // maximum Pt difference
  double nsigma_Pt_;      // pt uncertainty margin (in number of sigmas)
  double max_DCAMuMu_;    // DCA between the three muons
  double max_YTriplet_;   // |rapidity| of triplet
  const edm::InputTag theL3LinksLabel;                                //Needed to iterL3
  const edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;  //Needed to iterL3
};

#endif  //HLTMuonDimuonFilter_h
