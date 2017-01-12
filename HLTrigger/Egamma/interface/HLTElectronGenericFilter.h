#ifndef HLTElectronGenericFilter_h
#define HLTElectronGenericFilter_h

/** \class HLTElectronGenericFilter
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTElectronGenericFilter : public HLTFilter {

   public:
      explicit HLTElectronGenericFilter(const edm::ParameterSet&);
      ~HLTElectronGenericFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered electrons
      edm::InputTag varTag_; // input tag identifying product that contains the variable map
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
      edm::EDGetTokenT<reco::ElectronIsolationMap> varToken_;
      bool lessThan_;           // the cut is "<" or ">" ?
      double thrRegularEB_;     // threshold for regular cut (x < thr) - ECAL barrel
      double thrRegularEE_;     // threshold for regular cut (x < thr) - ECAL endcap
      double thrOverPtEB_;       // threshold for x/p_T < thr cut (isolations) - ECAL barrel
      double thrOverPtEE_;       // threshold for x/p_T < thr cut (isolations) - ECAL endcap
      double thrTimesPtEB_;      // threshold for x*p_T < thr cut (isolations) - ECAL barrel
      double thrTimesPtEE_;      // threshold for x*p_T < thr cut (isolations) - ECAL endcap
      int    ncandcut_;        // number of electrons required

      edm::InputTag l1EGTag_;
};

#endif //HLTElectronGenericFilter_h


