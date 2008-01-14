#ifndef HLTMuonL2PreFilter_h
#define HLTMuonL2PreFilter_h

/** \class HLTMuonL2PreFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

class HLTMuonL2PreFilter : public HLTFilter {

   public:
      explicit HLTMuonL2PreFilter(const edm::ParameterSet&);
      ~HLTMuonL2PreFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      bool triggeredByLevel1(reco::TrackRef& track,edm::Handle<L2MuonTrajectorySeedCollection> &museeds,std::vector<l1extra::L1MuonParticleRef>& vcands);
   private:
      edm::InputTag candTag_;  // input tag identifying product contains muons
      edm::InputTag linksTag_;  // input tag identifying product contains muons
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      double min_Pt_;           // pt threshold in GeV 
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)

};

#endif //HLTMuonL2PreFilter_h
