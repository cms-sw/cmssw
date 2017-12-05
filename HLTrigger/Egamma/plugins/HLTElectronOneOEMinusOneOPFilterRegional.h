#ifndef HLTElectronOneOEMinusOneOPFilterRegional_h
#define HLTElectronOneOEMinusOneOPFilterRegional_h

/** \class HLTElectronOneOEMinusOneOPFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTElectronOneOEMinusOneOPFilterRegional : public HLTFilter {

   public:
      explicit HLTElectronOneOEMinusOneOPFilterRegional(const edm::ParameterSet&);
      ~HLTElectronOneOEMinusOneOPFilterRegional() override;
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_; // input tag for the RecoCandidates from the previous filter
      edm::InputTag electronIsolatedProducer_;// input tag for the producer of electrons
      edm::InputTag electronNonIsolatedProducer_;// input tag for the producer of electrons
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
      edm::EDGetTokenT<reco::ElectronCollection> electronIsolatedToken_;
      edm::EDGetTokenT<reco::ElectronCollection> electronNonIsolatedToken_;
      bool doIsolated_;
      double barrelcut_; //  Eoverp barrel
      double endcapcut_; //  Eoverp endcap
      int    ncandcut_;        // number of electrons required
};

#endif //HLTElectronOneOEMinusOneOPFilterRegional_h
