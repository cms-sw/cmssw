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
#include "RecoMuon/MuonIsolation/interface/MuonIsolatorFactory.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTMuonIsoFilter : public HLTFilter {

   public:
      explicit HLTMuonIsoFilter(const edm::ParameterSet&);
      ~HLTMuonIsoFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      static bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &, const std::vector<reco::RecoChargedCandidateRef> &);

      edm::InputTag                                          candTag_;   // input tag identifying muon container
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> candToken_; // token identifying muon container
      edm::InputTag                                          previousCandTag_;   // input tag identifying product contains muons passing the previous level
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> previousCandToken_; // token identifying product contains muons passing the previous level
      std::vector<edm::InputTag>                                       depTag_;   // input tags identifying deposit maps
      std::vector<edm::EDGetTokenT<edm::ValueMap<reco::IsoDeposit> > > depToken_; // tokens identifying deposit maps
      edm::EDGetTokenT<edm::ValueMap<bool> > decMapToken_; // bool decision map

      const muonisolation::MuIsoBaseIsolator * theDepositIsolator;

      int    min_N_;          // minimum number of muons to fire the trigger
};

#endif //HLTMuonIsoFilter_h
