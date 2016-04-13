#ifndef HLTMuonL1TtoL3TkPreFilter_h
#define HLTMuonL1TtoL3TkPreFilter_h

/** \class HLTMuonL1TtoL3TkPreFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J-R Vlimant
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonL1TtoL3TkPreFilter : public HLTFilter {

   public:
      explicit HLTMuonL1TtoL3TkPreFilter(const edm::ParameterSet&);
      ~HLTMuonL1TtoL3TkPreFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      bool triggeredAtL1(const l1t::MuonRef & l1mu,std::vector<l1t::MuonRef>& vcands) const;

      edm::InputTag                    beamspotTag_ ;
      edm::EDGetTokenT<reco::BeamSpot> beamspotToken_ ;
      edm::InputTag                                          candTag_;   // input tag identifying product contains muons
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_; // token identifying product contains muons
      edm::InputTag                                          previousCandTag_;   // input tag identifying product contains muons passing the previous level
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_; // token identifying product contains muons passing the previous level
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      double min_Pt_;           // pt threshold in GeV
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
};

#endif //HLTMuonL1TtoL3TkPreFilter_h
