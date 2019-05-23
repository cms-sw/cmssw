#ifndef HLTFatJetMassFilter_h
#define HLTFatJetMassFilter_h

/** \class HLTFatJetMassFilter
 *
 *  \author Maurizio Pierini
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename jetType>
class HLTFatJetMassFilter : public HLTFilter {
public:
  explicit HLTFatJetMassFilter(const edm::ParameterSet&);
  ~HLTFatJetMassFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::EDGetTokenT<std::vector<jetType>> m_theJetToken;
  edm::InputTag inputJetTag_;  // input tag identifying jets
  double minMass_;
  double fatJetDeltaR_;
  double maxDeltaEta_;
  double maxJetEta_;
  double minJetPt_;
  int triggerType_;
};

#endif  //HLTFatJetMassFilter_h
