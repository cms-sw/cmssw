#ifndef HLTEgammaEtFilter_h
#define HLTEgammaEtFilter_h

/** \class HLTEgammaEtFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
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

class HLTEgammaEtFilter : public HLTFilter {
public:
  explicit HLTEgammaEtFilter(const edm::ParameterSet&);
  ~HLTEgammaEtFilter() override;
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputTag_;  // input tag identifying product contains egammas
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken_;
  double etcutEB_;    // Barrel Et threshold in GeV
  double etcutEE_;    // Endcap Et threshold in GeV
  double minEtaCut_;  // Min pseudorapidity cut
  double maxEtaCut_;  // Max pseudorapidity cut
  int ncandcut_;      // number of egammas required

  edm::InputTag l1EGTag_;
};

#endif  //HLTEgammaEtFilter_h
