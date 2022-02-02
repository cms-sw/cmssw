#ifndef HLTrigger_btau_HLTSumJetTagWithMatching_h
#define HLTrigger_btau_HLTSumJetTagWithMatching_h

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
#include "DataFormats/Math/interface/deltaR.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
class HLTSumJetTagWithMatching : public HLTFilter {
public:
  explicit HLTSumJetTagWithMatching(const edm::ParameterSet& config);
  ~HLTSumJetTagWithMatching() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;
  static float findTag(const T& jet, const reco::JetTagCollection& jetTags, float minDR);

private:
  const edm::InputTag m_Jets;     // input jet collection
  const edm::InputTag m_JetTags;  // input tag jet collection
  const edm::EDGetTokenT<std::vector<T> > m_JetsToken;
  const edm::EDGetTokenT<reco::JetTagCollection> m_JetTagsToken;
  const double m_MinTag;             // min tag requirement applied on the sum
  const double m_MaxTag;             // max tag requirement applied on the sum
  const unsigned int m_MinJetToSum;  // minimum number of jets to be considered in the sum
  const unsigned int m_MaxJetToSum;  // maximum number of jets to be considered in the sum
  const double m_deltaR;             // delta R condition for matching
  const bool m_UseMeanValue;         // use mean value instead of sum
  const int m_TriggerType;
};

#endif
