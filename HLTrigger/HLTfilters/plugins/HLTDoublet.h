#ifndef HLTrigger_HLTfilters_HLTDoublet_h
#define HLTrigger_HLTfilters_HLTDoublet_h

/** \class HLTDoublet
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a basic HLT
 *  trigger for pairs of object, evaluating all pairs with the first
 *  object from collection 1, and the second object from collection 2,
 *  cutting on variables relating to their 4-momentum representations.
 *  The object collections are assumed to be outputs of HLTSinglet
 *  single-object-type filters so that the access is thorugh
 *  RefToBases and polymorphic.
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

namespace trigger {
  class TriggerFilterObjectWithRefs;
}

template <typename T1, typename T2>
class HLTDoublet : public HLTFilter {
public:
  explicit HLTDoublet(const edm::ParameterSet&);
  ~HLTDoublet() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef std::vector<T2> T2Collection;
  typedef edm::Ref<T2Collection> T2Ref;

  // configuration
  const std::vector<edm::InputTag> originTag1_;  // input tag identifying originals 1st product
  const std::vector<edm::InputTag> originTag2_;  // input tag identifying originals 2nd product
  const edm::InputTag inputTag1_;                // input tag identifying filtered 1st product
  const edm::InputTag inputTag2_;                // input tag identifying filtered 2nd product
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken1_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken2_;
  const int triggerType1_;
  const int triggerType2_;
  const double min_Deta_, max_Deta_;    // Delta eta window
  const double min_Dphi_, max_Dphi_;    // Delta phi window
  const double min_DelR2_, max_DelR2_;  // Delta R^2 window
  const double min_Pt_, max_Pt_;        // Pt(1,2) window
  const double min_Minv_, max_Minv_;    // Minv(1,2) window
  const int min_N_;                     // number of pairs passing cuts required

  // calculated from configuration in c'tor
  // 1st and 2nd product are one and the same
  const bool same_;
  // cuts are on=true or off=false
  const bool cutdeta_, cutdphi_, cutdelr2_, cutpt_, cutminv_;
};

#endif  // HLTrigger_HLTfilters_HLTDoublet_h
