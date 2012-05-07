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

class HLTMuonDimuonL3Filter : public HLTFilter {

   public:
      explicit HLTMuonDimuonL3Filter(const edm::ParameterSet&);
      ~HLTMuonDimuonL3Filter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      bool triggeredByLevel2(const reco::TrackRef& track,std::vector<reco::RecoChargedCandidateRef>& vcands);

   private:
      edm::InputTag beamspotTag_ ;
      edm::InputTag candTag_;  // input tag identifying product contains muons
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      
      bool   fast_Accept_;      // flag to save time: stop processing after identification of the first valid pair
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
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
      double min_PtBalance_;    // minimum Pt difference
      double max_PtBalance_;    // maximum Pt difference
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
      double max_DzMuMu_;        // Delta z between the two muons
      double max_YPair_;        // |rapidity| of pair
      bool saveTag_;            // should we save the input collection ?

};

#endif //HLTMuonDimuonFilter_h
