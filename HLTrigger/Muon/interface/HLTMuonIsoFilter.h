#ifndef HLTMuonIsoFilter_h
#define HLTMuonIsoFilter_h

/** \class HLTMuonIsoFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing
 *  the isolation filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"

class HLTMuonIsoFilter : public HLTFilter {

   public:
      explicit HLTMuonIsoFilter(const edm::ParameterSet&);
      ~HLTMuonIsoFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &, const std::vector<reco::RecoChargedCandidateRef> &);
   private:
      edm::InputTag candTag_; // input tag identifying muon container
      edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
      std::vector<edm::InputTag> depTag_;  // input tag identifying deposit maps

      const muonisolation::MuIsoBaseIsolator * theDepositIsolator;

      int    min_N_;          // minimum number of muons to fire the trigger
      bool saveTags_;            // should we save the input collection ?
};

#endif //HLTMuonIsoFilter_h
