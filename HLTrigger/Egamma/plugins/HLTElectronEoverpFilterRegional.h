#ifndef HLTElectronEoverpFilterRegional_h
#define HLTElectronEoverpFilterRegional_h

/** \class HLTElectronEoverpFilterRegional
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

class HLTElectronEoverpFilterRegional : public HLTFilter {
public:
  explicit HLTElectronEoverpFilterRegional(const edm::ParameterSet&);
  ~HLTElectronEoverpFilterRegional() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  edm::EDGetTokenT<reco::ElectronCollection> electronIsolatedToken_;
  edm::EDGetTokenT<reco::ElectronCollection> electronNonIsolatedToken_;
  edm::InputTag candTag_;                      // input tag for the RecoCandidates from the previous filter
  edm::InputTag electronIsolatedProducer_;     // input tag for the producer of electrons
  edm::InputTag electronNonIsolatedProducer_;  // input tag for the producer of electrons
  bool doIsolated_;
  double eoverpbarrelcut_;  //  Eoverp barrel
  double eoverpendcapcut_;  //  Eoverp endcap
  int ncandcut_;            // number of electrons required
};

#endif  //HLTElectronEoverpFilterRegional_h
