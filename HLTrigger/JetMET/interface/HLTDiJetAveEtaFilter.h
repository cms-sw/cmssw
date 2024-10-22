#ifndef HLTDiJetAveEtaFilter_h
#define HLTDiJetAveEtaFilter_h

/** \class HLTDiJetAveEtaFilter
 *
 *  \author Tomasz Fruboes
 *    based on HLTDiJetAveFilter
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
class HLTDiJetAveEtaFilter : public HLTFilter {
public:
  explicit HLTDiJetAveEtaFilter(const edm::ParameterSet&);
  ~HLTDiJetAveEtaFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::EDGetTokenT<std::vector<T>> m_theJetToken;
  edm::InputTag inputJetTag_;  // input tag identifying jets
  double minPtJet_;
  double minPtAve_;
  //double minPtJet3_;
  double minDphi_;
  double tagEtaMin_;
  double tagEtaMax_;
  double probeEtaMin_;
  double probeEtaMax_;
  int triggerType_;
};

#endif  //HLTDiJetAveEtaFilter_h
