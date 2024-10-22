#ifndef HLTrigger_btau_HLTSumJetTag_h
#define HLTrigger_btau_HLTSumJetTag_h

#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

template <typename T>
class HLTSumJetTag : public HLTFilter {
public:
  explicit HLTSumJetTag(const edm::ParameterSet& config);
  ~HLTSumJetTag() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  bool findTagValueByMinDeltaR2(float& jetTagValue,
                                const T& jet,
                                const reco::JetTagCollection& jetTags,
                                float maxDeltaR2) const;

  const edm::InputTag m_Jets;     // input jet collection
  const edm::InputTag m_JetTags;  // input tag collection
  const edm::EDGetTokenT<std::vector<T>> m_JetsToken;
  const edm::EDGetTokenT<reco::JetTagCollection> m_JetTagsToken;
  const double m_MinTag;       // min tag value
  const double m_MaxTag;       // max tag value
  const int m_MinJetToSum;     // min number of jets to be considered in the sum
  const int m_MaxJetToSum;     // max number of jets to be considered in the sum
  const bool m_UseMeanValue;   // consider mean of jet tags instead of their sum
  const bool m_MatchByDeltaR;  // find jet-tag value by Delta-R matching
  const bool m_MaxDeltaR;      // max Delta-R to assign jet-tag to a jet via Delta-R matching
  const int m_TriggerType;     // type of TriggerObject in TriggerFilterObjectWithRefs
};

#endif
