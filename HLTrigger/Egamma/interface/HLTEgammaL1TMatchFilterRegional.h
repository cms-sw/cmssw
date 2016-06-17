#ifndef HLTEgammaL1TMatchFilterRegional_h
#define HLTEgammaL1TMatchFilterRegional_h

/** \class HLTEgammaL1TMatchFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

//#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
//#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTEgammaL1TMatchFilterRegional : public HLTFilter {

  public:
    explicit HLTEgammaL1TMatchFilterRegional(const edm::ParameterSet&);
    ~HLTEgammaL1TMatchFilterRegional();
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    edm::InputTag candIsolatedTag_;         // input tag identifying product contains egammas
    edm::InputTag l1EGTag_;           // input tag identifying product contains egammas
    edm::InputTag candNonIsolatedTag_;      // input tag identifying product contains egammas
    edm::InputTag l1JetsTag_;//EGamma can now be seeded by L1 Jet seeds (important for high energy) 
    edm::InputTag l1TausTag_;//EGamma can now be seeded by L1 Tau seeds (extremely important for high energy) 
    edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candIsolatedToken_;
    edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candNonIsolatedToken_;

    edm::InputTag L1SeedFilterTag_;
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> L1SeedFilterToken_;
    bool doIsolated_;
   
    int    ncandcut_;        // number of egammas required
    // L1 matching cuts
    double region_eta_size_;
    double region_eta_size_ecap_;
    double region_phi_size_;
    double barrel_end_;
    double endcap_end_;

  private:
    bool matchedToL1Cand(const std::vector<l1t::EGammaRef>& l1Cands,const float scEta,const float scPhi) const;
    bool matchedToL1Cand(const std::vector<l1t::JetRef>& l1Cands,const float scEta,const float scPhi) const;
    bool matchedToL1Cand(const std::vector<l1t::TauRef>& l1Cands,const float scEta,const float scPhi) const;
};

#endif //HLTEgammaL1TMatchFilterRegional_h
