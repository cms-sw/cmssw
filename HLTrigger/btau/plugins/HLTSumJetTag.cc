#include "HLTSumJetTag.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include <numeric>

template <typename T>
HLTSumJetTag<T>::HLTSumJetTag(const edm::ParameterSet& config)
    : HLTFilter(config),
      m_Jets(config.getParameter<edm::InputTag>("Jets")),
      m_JetTags(config.getParameter<edm::InputTag>("JetTags")),
      m_JetsToken(consumes(m_Jets)),
      m_JetTagsToken(consumes(m_JetTags)),
      m_MinTag(config.getParameter<double>("MinTag")),
      m_MaxTag(config.getParameter<double>("MaxTag")),
      m_MinJetToSum(config.getParameter<int>("MinJetToSum")),
      m_MaxJetToSum(config.getParameter<int>("MaxJetToSum")),
      m_UseMeanValue(config.getParameter<bool>("UseMeanValue")),
      m_MatchByDeltaR(config.getParameter<bool>("MatchByDeltaR")),
      m_MaxDeltaR(config.getParameter<double>("MaxDeltaR")),
      m_TriggerType(config.getParameter<int>("TriggerType")) {
  // MaxDR must be positive if matching-by-DR option is enabled
  if (m_MatchByDeltaR and m_MaxDeltaR <= 0) {
    throw cms::Exception("HLTSumJetTag") << "invalid value for parameter \"MaxDeltaR\" (must be > 0): " << m_MaxDeltaR;
  }
  // Min must be smaller or equal than Max (unless Max<0)
  if (m_MaxJetToSum >= 0 and m_MinJetToSum > m_MaxJetToSum) {
    throw cms::Exception("HLTSumJetTag")
        << "invalid value for min/max number of jets to sum: parameter \"MinJetToSum\" (" << m_MinJetToSum
        << ") must be <= than \"MaxJetToSum\" (" << m_MaxJetToSum << ") if \"MaxJetToSum\" >= 0";
  }

  edm::LogInfo("") << " (HLTSumJetTag) trigger cuts: \n"
                   << " \ttype of jets used: " << m_Jets.encode() << " \n"
                   << " \ttype of tagged jets used: " << m_JetTags.encode() << " \n"
                   << " \tmin/max tag value: [" << m_MinTag << ", " << m_MaxTag << "] \n"
                   << " \tmin/max number of jets to sum: [" << m_MinJetToSum << ", " << m_MaxJetToSum << "] \n"
                   << " \tuse mean value of jet tags: " << m_UseMeanValue << " \n"
                   << " \tassign jet-tag values by Delta-R matching: " << m_MatchByDeltaR << "\n"
                   << " \tmax Delta-R for jet-tag assignment by Delta-R matching: " << m_MaxDeltaR << "\n"
                   << " \tTriggerType: " << m_TriggerType << " \n";
}

template <typename T>
void HLTSumJetTag<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("Jets", edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("JetTags", edm::InputTag("hltJetTagCollection"));
  desc.add<double>("MinTag", 0.);
  desc.add<double>("MaxTag", 999999.0);
  desc.add<int>("MinJetToSum", 1);
  desc.add<int>("MaxJetToSum", -1);
  desc.add<bool>("UseMeanValue", true);
  desc.add<bool>("MatchByDeltaR", false);
  desc.add<double>("MaxDeltaR", 0.4);
  desc.add<int>("TriggerType", 86);
  descriptions.add(defaultModuleLabel<HLTSumJetTag<T>>(), desc);
}

template <typename T>
bool HLTSumJetTag<T>::hltFilter(edm::Event& event,
                                const edm::EventSetup& setup,
                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(m_Jets);

  typedef edm::Ref<std::vector<T>> TRef;

  auto const h_Jets = event.getHandle(m_JetsToken);
  auto const h_JetTags = event.getHandle(m_JetTagsToken);

  std::vector<TRef> jetRefCollection;
  jetRefCollection.reserve(h_Jets->size());

  std::vector<float> jetTagValues;
  jetTagValues.reserve(h_Jets->size());

  if (m_MaxJetToSum == 0) {
    // return false in case max jet is set to zero --> because no sum can be computed
    LogDebug("HLTSumJetTag") << "Parameter \"MaxJetToSum\" set to zero --> Return False";
    return false;
  }

  // save jetTagValues associated to each jet
  auto const maxDeltaR2 = m_MaxDeltaR * m_MaxDeltaR;
  for (size_t iJet = 0; iJet < h_Jets->size(); ++iJet) {
    TRef const jetRef(h_Jets, iJet);

    float jetTag = -1;
    if (m_MatchByDeltaR) {
      // if match-by-DR is used, update jetTag by reference and use it only if the matching is valid
      if (not findTagValueByMinDeltaR2(jetTag, (*h_Jets)[iJet], *h_JetTags, maxDeltaR2)) {
        continue;
      }
    } else {
      // operator[] checks consistency between h_Jets and h_JetTags
      jetTag = (*h_JetTags)[reco::JetBaseRef(jetRef)];
    }

    jetRefCollection.emplace_back(jetRef);
    jetTagValues.emplace_back(jetTag);

    LogDebug("HLTSumJetTag") << "Input Jets -- Jet[" << iJet << "] (id = " << jetRefCollection.back().id() << ")"
                             << ": tag=" << jetTag << ", pt=" << jetRefCollection.back()->pt()
                             << ", eta=" << jetRefCollection.back()->eta()
                             << ", phi=" << jetRefCollection.back()->phi();
  }

  // sorting from largest to smaller
  std::vector<size_t> jetTagSortedIndices(jetTagValues.size());
  std::iota(jetTagSortedIndices.begin(), jetTagSortedIndices.end(), 0);
  std::sort(jetTagSortedIndices.begin(), jetTagSortedIndices.end(), [&](const size_t& i1, const size_t& i2) {
    return jetTagValues[i1] > jetTagValues[i2];
  });

  // consider only up to m_MaxJetToSum jet-tag indexes if max is set to be positive
  if (m_MaxJetToSum > 0 and int(jetTagSortedIndices.size()) > m_MaxJetToSum)
    jetTagSortedIndices.resize(m_MaxJetToSum);

  // sum jet tags and possibly take mean value
  float sumJetTag = 0;
  for (auto const idx : jetTagSortedIndices) {
    sumJetTag += jetTagValues[idx];
    LogDebug("HLTSumJetTag") << "Selected Jets -- Jet[" << idx << "] (id = " << jetRefCollection[idx].id() << ")"
                             << ": tag=" << jetTagValues[idx] << ", pt=" << jetRefCollection[idx]->pt()
                             << ", eta=" << jetRefCollection[idx]->eta() << ", phi=" << jetRefCollection[idx]->phi();
  }

  if (m_UseMeanValue and not jetTagSortedIndices.empty()) {
    sumJetTag /= jetTagSortedIndices.size();
  }

  // filter decision: (number of tagged jets >= min) and (minTag <= SumTag <= maxTag)
  auto const accept =
      (int(jetTagSortedIndices.size()) >= m_MinJetToSum and m_MinTag <= sumJetTag and sumJetTag <= m_MaxTag);

  LogDebug("HLTSumJetTag") << "Filter Result = " << accept << " (SumJetTag = " << sumJetTag
                           << ", Num. of Jets = " << jetTagSortedIndices.size()
                           << ") [UseMeanValue = " << m_UseMeanValue << "]";

  // if accept, fill TriggerFilterObjectWithRefs
  if (accept) {
    for (auto const idx : jetTagSortedIndices) {
      filterproduct.addObject(m_TriggerType, jetRefCollection[idx]);
    }
  }

  return accept;
}

template <typename T>
bool HLTSumJetTag<T>::findTagValueByMinDeltaR2(float& jetTagValue,
                                               const T& jet,
                                               const reco::JetTagCollection& jetTags,
                                               float maxDeltaR2) const {
  bool ret = false;
  for (const auto& jetTag : jetTags) {
    auto const tmpDR2 = reco::deltaR2(jet, *(jetTag.first));
    if (tmpDR2 < maxDeltaR2) {
      maxDeltaR2 = tmpDR2;
      jetTagValue = jetTag.second;
      ret = true;
    }
  }

  return ret;
}
