#ifndef HLTrigger_btau_HLTJetTag_h
#define HLTrigger_btau_HLTJetTag_h

/** \class HLTJetTag
 *
 *  This class is an HLTFilter (a spcialized EDFilter) implementing
 *  tagged multi-jet trigger for b and tau.
 *  It should be run after the normal multi-jet trigger.
 *
 *
 *  \author Arnaud Gay, Ian Tomalin
 *  \maintainer Andrea Bocci
 *
 */

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename T>
class HLTJetTag : public HLTFilter {
public:
  explicit HLTJetTag(const edm::ParameterSet& config);
  ~HLTJetTag() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag m_Jets;  // module label of input JetCollection
  edm::EDGetTokenT<std::vector<T> > m_JetsToken;
  edm::InputTag m_JetTags;  // module label of input JetTagCollection
  edm::EDGetTokenT<reco::JetTagCollection> m_JetTagsToken;
  double m_MinTag, m_MaxTag;  // tag descriminator cuts applied to each jet
  int m_MinJets;              // min. number of jets required to be tagged
  int m_TriggerType;
};

#endif  // HLTrigger_btau_HLTJetTag_h
