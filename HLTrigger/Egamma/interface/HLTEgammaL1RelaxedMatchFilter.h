#ifndef HLTEgammaL1RelaxedMatchFilter_h
#define HLTEgammaL1RelaxedMatchFilter_h

/** \class HLTEgammaL1RelaxedMatchFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaL1RelaxedMatchFilter : public HLTFilter {

   public:
      explicit HLTEgammaL1RelaxedMatchFilter(const edm::ParameterSet&);
      ~HLTEgammaL1RelaxedMatchFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains egammas
      edm::InputTag l1IsolTag_; 
      edm::InputTag l1NonIsolTag_;
      int    ncandcut_;        // number of egammas required
      // L1 matching cuts
      double region_eta_size_;
      double region_eta_size_ecap_;
      double region_phi_size_;
      double barrel_end_;
      double endcap_end_;
};

#endif //HLTEgammaL1RelaxedMatchFilter_h
