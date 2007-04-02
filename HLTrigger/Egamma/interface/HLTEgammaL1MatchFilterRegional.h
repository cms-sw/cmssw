#ifndef HLTEgammaL1MatchFilterRegional_h
#define HLTEgammaL1MatchFilterRegional_h

/** \class HLTEgammaL1MatchFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaL1MatchFilterRegional : public HLTFilter {

   public:
      explicit HLTEgammaL1MatchFilterRegional(const edm::ParameterSet&);
      ~HLTEgammaL1MatchFilterRegional();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag l1IsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag candNonIsolatedTag_; // input tag identifying product contains egammas
      edm::InputTag l1NonIsolatedTag_; // input tag identifying product contains egammas
      bool doIsolated_;

      int    ncandcut_;        // number of egammas required
      // L1 matching cuts
      double region_eta_size_;
      double region_eta_size_ecap_;
      double region_phi_size_;
      double barrel_end_;
      double endcap_end_;
};

#endif //HLTEgammaL1MatchFilterRegional_h
