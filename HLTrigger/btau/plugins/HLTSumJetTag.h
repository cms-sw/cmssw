#ifndef HLTrigger_btau_HLTSumJetTag_h
#define HLTrigger_btau_HLTSumJetTag_h

#include <string>
#include <vector>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
class HLTSumJetTag : public HLTFilter {
public:
  explicit HLTSumJetTag(const edm::ParameterSet& config);
  ~HLTSumJetTag() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag m_Jets;     // input jet collection
  edm::InputTag m_JetTags;  // input tag collection
  const edm::EDGetTokenT<std::vector<T> > m_JetsToken;
  const edm::EDGetTokenT<reco::JetTagCollection> m_JetTagsToken;
  double m_MinTag;             // min tag value
  double m_MaxTag;             // max tag value
  unsigned int m_MinJetToSum;  // min number of jets to be considered in the mean
  unsigned int m_MaxJetToSum;  // max number of jets to be considered in the mean
  bool m_UseMeanValue;         // consider mean instead of sum of jet tags
  int m_TriggerType;
};

#endif
