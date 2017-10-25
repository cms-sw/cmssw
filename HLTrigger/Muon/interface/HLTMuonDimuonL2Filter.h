#ifndef HLTMuonDimuonL2Filter_h
#define HLTMuonDimuonL2Filter_h

/** \class HLTMuonDimuonL2Filter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a muon pair
 *  filter for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "HLTrigger/Muon/interface/HLTMuonL2ToL1Map.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonDimuonL2Filter : public HLTFilter {

   public:
      explicit HLTMuonDimuonL2Filter(const edm::ParameterSet&);
      ~HLTMuonDimuonL2Filter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      edm::InputTag                    beamspotTag_ ;
      edm::EDGetTokenT<reco::BeamSpot> beamspotToken_ ;
      edm::InputTag                                          candTag_;   // input tag identifying product contains muons
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_; // token identifying product contains muons
      edm::InputTag                                          previousCandTag_;   // input tag identifying product contains muons passing the previous level
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_; // token identifying product contains muons passing the previous level
      /// input tag of the map from the L2 seed to the sister L2 seeds of cleaned tracks
      edm::InputTag             seedMapTag_;
      edm::EDGetTokenT<SeedMap> seedMapToken_;

      bool   fast_Accept_;      // flag to save time: stop processing after identification of the first valid pair
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      int    min_Nstations_;    // threshold on number of valid stations for muon
      int    min_Nchambers_;    // threshold on number of valid chambers for muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      int    chargeOpt_;        // Charge option (0:nothing; +1:same charge, -1:opposite charge)
      double min_PtPair_;       // minimum Pt for the dimuon system
      double min_PtMax_;        // minimum Pt for muon with max Pt in pair
      double min_PtMin_;        // minimum Pt for muon with min Pt in pair
      double min_InvMass_;      // minimum invariant mass of pair
      double max_InvMass_;      // maximum invariant mass of pair
      double min_Acop_;         // minimum acoplanarity
      double max_Acop_;         // maximum acoplanarity
      double min_Angle_;        // minimum 3D angle
      double max_Angle_;        // maximum 3D angle
      double min_PtBalance_;    // minimum Pt difference
      double max_PtBalance_;    // maximum Pt difference
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)

};

#endif //HLTMuonDimuonFilter_h
