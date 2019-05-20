#ifndef HLTDiJetAveFilter_h
#define HLTDiJetAveFilter_h

/** \class HLTDiJetAveFilter
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename T>
class HLTDiJetAveFilter : public HLTFilter {
public:
  explicit HLTDiJetAveFilter(const edm::ParameterSet&);
  ~HLTDiJetAveFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::EDGetTokenT<std::vector<T>> m_theJetToken;
  edm::InputTag inputJetTag_;  // input tag identifying jets
  double minPtAve_;
  double minPtJet3_;
  double minDphi_;
  int triggerType_;
};

#endif  //HLTDiJetAveFilter_h
