#ifndef HLTEgammaGenericQuadraticFilter_h
#define HLTEgammaGenericQuadraticFilter_h

/** \class HLTEgammaGenericQuadraticFilter
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTEgammaGenericQuadraticFilter : public HLTFilter {

   public:
      explicit HLTEgammaGenericQuadraticFilter(const edm::ParameterSet&);
      ~HLTEgammaGenericQuadraticFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered candidates
      edm::InputTag varTag_; // input tag identifying product that contains the variable map
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
      edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> varToken_;

      bool lessThan_;           // the cut is "<" or ">" ?
      bool useEt_;              // use E or Et in relative isolation cuts
/*  Barrel quadratic threshold function:
      vali (<= or >=) thrRegularEB_ + (E or Et)*thrOverEEB_ + (E or Et)*(E or Et)*thrOverE2EB_
    Endcap quadratic threshold function:
      vali (<= or >=) thrRegularEE_ + (E or Et)*thrOverEEE_ + (E or Et)*(E or Et)*thrOverE2EE_
*/
      double thrRegularEB_;     // threshold value for zeroth order term - ECAL barrel
      double thrRegularEE_;     // threshold value for zeroth order term - ECAL endcap
      double thrOverEEB_;       // coefficient for first order term - ECAL barrel
      double thrOverEEE_;       // coefficient for first order term - ECAL endcap
      double thrOverE2EB_;      // coefficient for second order term - ECAL barrel
      double thrOverE2EE_;      // coefficient for second order term - ECAL endcap
      int    ncandcut_;        // number of candidates required

      edm::InputTag l1EGTag_;
};

#endif //HLTEgammaGenericQuadraticFilter_h


