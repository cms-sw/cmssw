#ifndef HLTElectronMissingHitsFilter_h
#define HLTElectronMissingHitsFilter_h

/** \class HLTElectronMissingHitsFilter
 *
 *  \author Matteo Sani (UCSD)
 */

namespace edm {
  class ConfigurationDescriptions;
}

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

class HLTElectronMissingHitsFilter : public HLTFilter {
public:
  explicit HLTElectronMissingHitsFilter(const edm::ParameterSet&);
  ~HLTElectronMissingHitsFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // input tag and token for the RecoCandidates from the previous filter
  edm::InputTag candTag_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  // input tag and token for the producer of electrons
  edm::InputTag electronTag_;
  edm::EDGetTokenT<reco::ElectronCollection> electronToken_;

  int barrelcut_;  // barrel cut
  int endcapcut_;  // endcap cut
  int ncandcut_;   // number of electrons required
};

#endif
