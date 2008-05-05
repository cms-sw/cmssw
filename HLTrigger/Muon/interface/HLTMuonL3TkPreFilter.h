#ifndef HLTMuonL3TkPreFilter_h
#define HLTMuonL3TkPreFilter_h

/** \class HLTMuonL3TkPreFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a first
 *  filtering for HLT muons
 *
 *  \author J-R Vlimant
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

class HLTMuonL3TkPreFilter : public HLTFilter {

   public:
      explicit HLTMuonL3TkPreFilter(const edm::ParameterSet&);
      ~HLTMuonL3TkPreFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      bool triggeredAtL2(const reco::TrackRef& staTrack, std::vector<reco::RecoChargedCandidateRef>& vcands);
   private:

      edm::InputTag beamspotTag_ ;
      edm::InputTag candTag_;  // input tag identifying product contains muons
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      int    min_N_;            // minimum number of muons to fire the trigger
      double max_Eta_;          // Eta cut
      int    min_Nhits_;        // threshold on number of hits on muon
      double max_Dr_;           // impact parameter cut
      double max_Dz_;           // dz cut
      double min_Pt_;           // pt threshold in GeV 
      double nsigma_Pt_;        // pt uncertainty margin (in number of sigmas)
      bool saveTag_;            // should we save the input collection ?
};

#endif //HLTMuonL3TkPreFilter_h
