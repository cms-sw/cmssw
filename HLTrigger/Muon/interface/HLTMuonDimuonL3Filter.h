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
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonDimuonL3Filter : public HLTFilter {

   public:
      explicit HLTMuonDimuonL3Filter(const edm::ParameterSet&);
      ~HLTMuonDimuonL3Filter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      static bool triggeredByLevel2(reco::TrackRef const & track, std::vector<reco::RecoChargedCandidateRef> const & vcands);

      edm::InputTag                    beamspotTag_ ;
      edm::EDGetTokenT<reco::BeamSpot> beamspotToken_ ;
      edm::InputTag                                          candTag_;   // input tag identifying product contains muons
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_; // token identifying product contains muons
      edm::InputTag                                          previousCandTag_;   // input tag identifying product contains muons passing the previous level
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_; // tokenidentifying product contains muons passing the previous level
      bool   previousCandIsL2_;
      bool   fast_Accept_;      // flag to save time: stop processing after identification of the first valid pair
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      int    chargeOpt_;        // Charge option (0:nothing; +1:same charge, -1:opposite charge)
      std::vector<double> min_PtPair_;       // minimum Pt for the dimuon system
      std::vector<double> max_PtPair_;       // miaximum Pt for the dimuon system
      std::vector<double> min_PtMax_;        // minimum Pt for muon with max Pt in pair
      std::vector<double> min_PtMin_;        // minimum Pt for muon with min Pt in pair
      std::vector<double> max_PtMin_;        // maximum Pt for muon with min Pt in pair
      std::vector<double> min_InvMass_;      // minimum invariant mass of pair
      std::vector<double> max_InvMass_;      // maximum invariant mass of pair
      double min_Acop_;         // minimum acoplanarity
      double max_Acop_;         // maximum acoplanarity
      double min_PtBalance_;    // minimum Pt difference
      double max_PtBalance_;    // maximum Pt difference
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
      double max_DCAMuMu_;      // DCA between the two muons
      double max_YPair_;        // |rapidity| of pair
      bool cutCowboys_;             ///< if true, reject muon-track pairs that bend towards each other
};

#endif //HLTMuonDimuonFilter_h
