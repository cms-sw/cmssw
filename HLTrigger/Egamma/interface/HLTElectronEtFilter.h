#ifndef HLTElectronEtFilter_h
#define HLTElectronEtFilter_h

/** \class HLTElectronEtFilter
 *
 *  \author Alessio Ghezzi
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTElectronEtFilter : public HLTFilter {

   public:
      explicit HLTElectronEtFilter(const edm::ParameterSet&);
      ~HLTElectronEtFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_; // input tag identifying product that contains filtered electrons
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;

      double EtEB_;     // threshold for regular cut (x < thr) - ECAL barrel
      double EtEE_;     // threshold for regular cut (x < thr) - ECAL endcap

      bool doIsolated_;

      edm::InputTag L1IsoCollTag_;
      edm::InputTag L1NonIsoCollTag_;
      int ncandcut_;
};

#endif //HLTElectronEtFilter_h


