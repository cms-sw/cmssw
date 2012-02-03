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

namespace edm {
   class ConfigurationDescriptions;
}

class HLTMuonL3PreFilter : public HLTFilter {

   public:
      explicit HLTMuonL3PreFilter(const edm::ParameterSet&);
      ~HLTMuonL3PreFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      bool triggeredByLevel2(const reco::TrackRef& track,std::vector<reco::RecoChargedCandidateRef>& vcands);
   private:

      edm::InputTag beamspotTag_ ;
      edm::InputTag candTag_;  // input tag identifying product contains muons
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // maximum impact parameter cut
      double min_Dr_;           // minimum impact parameter cut
      double max_Dz_;           // dz cut
      double min_DxySig_;       // dxy significance cut
      double min_Pt_;           // pt threshold in GeV 
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
      double max_NormalizedChi2_; // cutoff in normalized chi2
      double max_DXYBeamSpot_; // cutoff in dxy from the beamspot
  int min_NmuonHits_; // cutoff in minumum number of chi2 hits
  double max_PtDifference_; // cutoff in maximum different between global track and tracker track
  double min_TrackPt_; //cutoff in tracker track pt

  bool devDebug_;
  

};

#endif //HLTMuonL3PreFilter_h
