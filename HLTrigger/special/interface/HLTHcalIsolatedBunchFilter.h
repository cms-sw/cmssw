#ifndef HLTHcalIsolatedBunchFilter_h
#define HLTHcalIsolatedBunchFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTHcalIsolatedBunchFilter : public HLTFilter {

public:
  explicit HLTHcalIsolatedBunchFilter(const edm::ParameterSet&);
  ~HLTHcalIsolatedBunchFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::InputTag hltTauSeedLabel_, hltJetSeedLabel_;
  double minEta_, maxEta_;
  double minPhi_, maxPhi_;
  double minPt_;
  edm::EDGetTokenT<l1t::JetBxCollection>   hltJetToken_;
  edm::EDGetTokenT<l1t::TauBxCollection>   hltTauToken_;
};

#endif
