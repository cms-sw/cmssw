#ifndef HLTEgammaEtaFilter_h
#define HLTEgammaEtaFilter_h

/** \class HLTEgammaEtaFilter
 *
 *  \author Abanti Ranadhir Sahasransu (VUB, Belgium)
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

class HLTEgammaEtaFilter : public HLTFilter {
public:
  explicit HLTEgammaEtaFilter(const edm::ParameterSet&);
  ~HLTEgammaEtaFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputTag_;  // input tag identifying product contains egammas
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken_;
  double minEtacut_;  // Min Eta threshold
  double maxEtacut_;  // Max Eta threshold
  int ncandcut_;      // number of egammas required

  edm::InputTag l1EGTag_;
};

#endif  //HLTEgammaEtaFilter_h
