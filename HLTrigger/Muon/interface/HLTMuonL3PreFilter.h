#ifndef HLTMuonL3PreFilter_h
#define HLTMuonL3PreFilter_h

/** \class HLTMuonL3PreFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J. Alcaraz, J-R Vlimant
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


class HLTMuonL3PreFilter : public HLTFilter {
   public:
      explicit HLTMuonL3PreFilter(const edm::ParameterSet&);
      ~HLTMuonL3PreFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      bool triggeredByLevel2(const reco::TrackRef& track,std::vector<reco::RecoChargedCandidateRef>& vcands) const;
      const edm::InputTag                    beamspotTag_ ;
      const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_ ;
      const edm::InputTag                                          candTag_;   // input tag identifying product contains muons
      const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_; // token identifying product contains muons
      const edm::InputTag                                          previousCandTag_;   // input tag identifying product contains muons passing the previous level
      const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_; // token identifying product contains muons passing the previous level
      const int    min_N_;            // minimum number of muons to fire the trigger
      const double max_Eta_;          // Eta cut
      const int    min_Nhits_;        // threshold on number of hits on muon
      const double max_Dr_;           // maximum impact parameter cut
      const double min_Dr_;           // minimum impact parameter cut
      const double max_Dz_;           // dz cut
      const double min_DxySig_;       // dxy significance cut
      const double min_Pt_;           // pt threshold in GeV
      const double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
      const double max_NormalizedChi2_; // cutoff in normalized chi2
      const double max_DXYBeamSpot_;    // cutoff in dxy from the beamspot
      const double min_DXYBeamSpot_;    // minimum cut on dxy from the beamspot
      const int min_NmuonHits_;         // cutoff in minumum number of chi2 hits
      const double max_PtDifference_;   // cutoff in maximum different between global track and tracker track
      const double min_TrackPt_;        // cutoff in tracker track pt
      const bool devDebug_;
      const edm::InputTag theL3LinksLabel;
      const edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;
};

#endif //HLTMuonL3PreFilter_h
