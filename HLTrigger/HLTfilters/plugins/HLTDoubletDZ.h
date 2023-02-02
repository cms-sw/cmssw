#ifndef HLTrigger_HLTfilters_HLTDoubletDZ_h
#define HLTrigger_HLTfilters_HLTDoubletDZ_h

//
// Class implements |dZ|<Max for a pair of two objects
//

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <string>
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

namespace trigger {
  class TriggerFilterObjectWithRefs;
}

template <typename T1, typename T2>
class HLTDoubletDZ : public HLTFilter {
public:
  explicit HLTDoubletDZ(edm::ParameterSet const&);
  ~HLTDoubletDZ() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool hltFilter(edm::Event& iEvent,
                 edm::EventSetup const& iSetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  using T1Ref = edm::Ref<std::vector<T1>>;
  using T2Ref = edm::Ref<std::vector<T2>>;

  bool getCollections(edm::Event const& iEvent,
                      std::vector<T1Ref>& coll1,
                      std::vector<T2Ref>& coll2,
                      trigger::TriggerFilterObjectWithRefs& filterproduct) const;

  bool haveSameSuperCluster(T1 const& c1, T2 const& c2) const;

  bool passCutMinDeltaR(T1 const& c1, T2 const& c2) const;

  bool computeDZ(edm::Event const& iEvent, T1 const& c1, T2 const& c2) const;

  // configuration
  const std::vector<edm::InputTag> originTag1_;  // input tag identifying originals 1st product
  const std::vector<edm::InputTag> originTag2_;  // input tag identifying originals 2nd product
  const edm::InputTag inputTag1_;                // input tag identifying filtered 1st product
  const edm::InputTag inputTag2_;                // input tag identifying filtered 2nd product
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken1_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken2_;
  const edm::EDGetTokenT<reco::ElectronCollection> electronToken_;
  const int triggerType1_;
  const int triggerType2_;
  const double minDR_;         // minimum dR between two objects to be considered a pair
  const double minDR2_;        // minDR_ * minDR_
  const double maxDZ_;         // number of pairs passing cuts required
  const int min_N_;            // number of pairs passing cuts required
  const int minPixHitsForDZ_;  // minimum number of required pixel hits to check DZ
  const bool checkSC_;         // make sure SC constituents are different
  const bool same_;            // 1st and 2nd product are one and the same
};

#endif  // HLTrigger_HLTfilters_HLTDoubletDZ_h
