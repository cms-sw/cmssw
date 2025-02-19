#ifndef HLTEgammaL1MatchFilterRegional_h
#define HLTEgammaL1MatchFilterRegional_h

/** \class HLTEgammaL1MatchFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

//
// class decleration
//

class HLTEgammaL1MatchFilterRegional : public HLTFilter {

   public:
      explicit HLTEgammaL1MatchFilterRegional(const edm::ParameterSet&);
      ~HLTEgammaL1MatchFilterRegional();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag candIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag l1IsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag candNonIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag l1NonIsolatedTag_; // input tag identifying product contains egammas

      edm::InputTag L1SeedFilterTag_;
      bool doIsolated_;

      int    ncandcut_;        // number of egammas required
      // L1 matching cuts
      double region_eta_size_;
      double region_eta_size_ecap_;
      double region_phi_size_;
      double barrel_end_;
      double endcap_end_;

 public:
      bool matchedToL1Cand(const std::vector<l1extra::L1EmParticleRef >& l1Cands,const float scEta,const float scPhi);

};

#endif //HLTEgammaL1MatchFilterRegional_h
