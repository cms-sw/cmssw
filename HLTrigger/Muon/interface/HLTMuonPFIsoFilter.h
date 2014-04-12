#ifndef HLTMuonPFIsoFilter_h
#define HLTMuonPFIsoFilter_h

/** \class HLTMuonPFIsoFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing
 *  the isolation filtering for HLT muons
 *
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

class HLTMuonPFIsoFilter : public HLTFilter {

   public:
      explicit HLTMuonPFIsoFilter(const edm::ParameterSet&);
      ~HLTMuonPFIsoFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      static bool triggerdByPreviousLevel(const reco::RecoChargedCandidateRef &, const std::vector<reco::RecoChargedCandidateRef> &);

      edm::InputTag                                          		candTag_          ;   // input tag identifying muon container
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> 		candToken_        ;   // token identifying muon container
      edm::InputTag                                          		previousCandTag_  ;   // input tag identifying product contains muons passing the previous level
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> 		previousCandToken_;   // token identifying product contains muons passing the previous level
      std::vector<edm::InputTag>                                    depTag_           ;   // input tags identifying deposit maps
      std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > 		depToken_         ;   // tokens identifying deposit maps
      edm::InputTag 												rhoTag_			  ;   // input tag identifying rho container
      edm::EDGetTokenT<double> 										rhoToken_         ;   // token identifying rho container

      double maxIso_  		;       // max PF iso deposit allowed
      int    min_N_   		;       // minimum number of muons to fire the trigger
      bool   onlyCharged_	; 		// if true, only consider the charged component for isolation
      bool   doRho_  		; 		// if true, apply deltaBeta correction
      double effArea_  		;       // value for effective area for rho correction
};

#endif //HLTMuonPFIsoFilter_h
