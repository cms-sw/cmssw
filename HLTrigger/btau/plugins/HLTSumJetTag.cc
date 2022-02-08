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
      m_JetsToken(consumes<std::vector<T>>(m_Jets)),
      m_JetTagsToken(consumes<reco::JetTagCollection>(m_JetTags)),
      m_MinTag(config.getParameter<double>("MinTag")),
      m_MaxTag(config.getParameter<double>("MaxTag")),
      m_MinJetToSum(config.getParameter<unsigned int>("MinJetToSum")),
      m_MaxJetToSum(config.getParameter<unsigned int>("MaxJetToSum")),
      m_UseMeanValue(config.getParameter<bool>("UseMeanValue")),
      m_MatchByDeltaR(config.getParameter<bool>("MatchByDeltaR")),
      m_MaxDeltaR(config.getParameter<double>("MaxDeltaR")),
      m_TriggerType(config.getParameter<int>("TriggerType")) {

  if (m_MatchByDeltaR and m_MaxDeltaR <= 0) {
    throw cms::Exception("HLTSumJetTag") << "invalid value for parameter \"MaxDeltaR\" (must be > 0): " << m_MaxDeltaR;
  }

  edm::LogInfo("") << " (HLTSumJetTag) trigger cuts: \n"
                   << " \ttype of jets used: " << m_Jets.encode() << " \n"
                   << " \ttype of tagged jets used: " << m_JetTags.encode() << " \n"
                   << " \tmin/max tag value: [" << m_MinTag << ".." << m_MaxTag << "] \n"
                   << " \tmin/max number of jets to sum: [" << m_MinJetToSum << ".." << m_MaxJetToSum << "] \n"
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
  desc.add<unsigned int>("MinJetToSum", 1);
  desc.add<unsigned int>("MaxJetToSum", 99);
  desc.add<bool>("UseMeanValue", true);
  desc.add<bool>("MatchByDeltaR", false);
  desc.add<double>("MaxDeltaR", 0.4);
  desc.add<int>("TriggerType", 86);
  descriptions.add(defaultModuleLabel<HLTSumJetTag<T>>(), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
bool HLTSumJetTag<T>::hltFilter(edm::Event& event,
                                const edm::EventSetup& setup,
                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {

  if (saveTags())
    filterproduct.addCollectionTag(m_Jets);


  typedef edm::Ref<std::vector<T>> TRef;

  edm::Handle<std::vector<T> > h_JetsH;
  event.getByToken(m_JetsToken, h_JetsH);
  const std::vector<T> h_Jets = *h_JetsH;

  edm::Handle<reco::JetTagCollection> h_JetTagsH;
  event.getByToken(m_JetTagsToken, h_JetTagsH);
  const reco::JetTagCollection h_JetTags = *h_JetTagsH;

  std::vector<TRef> jetRefCollection;
  jetRefCollection.reserve(h_Jets.size());

  std::vector<float> jetTagValues;
  jetTagValues.reserve(h_Jets.size());

  //// Loop on jet tags and store values
  float maxDeltaR2 = m_MaxDeltaR * m_MaxDeltaR;
  for (size_t iJet = 0; iJet < h_Jets.size(); ++iJet) {
    jetRefCollection.emplace_back(h_JetsH, iJet);
    float jetTag = m_MatchByDeltaR ? findTagValueByMinDeltaR2(h_Jets[iJet], h_JetTags, maxDeltaR2)
      : h_JetTags[reco::JetBaseRef(jetRefCollection.back())];    
    jetTagValues.emplace_back(jetTag);
    
    LogDebug("HLTSumJetTag") << "Input Jets -- Jet[" << iJet << "] (id = " << jetRefCollection.back().id()
			     << "): tag=" << jetTag << ", pt=" << jetRefCollection.back()->pt()
			     << ", eta=" << jetRefCollection.back()->eta()
			     << ", phi=" << jetRefCollection.back()->phi();
  }

  // sorting from largest to smaller
  std::vector<size_t> jetTagSortedIndices(jetTagValues.size());
  std::iota(jetTagSortedIndices.begin(), jetTagSortedIndices.end(), 0);
  std::sort(jetTagSortedIndices.begin(), jetTagSortedIndices.end(), [&](const size_t & i1, const size_t & i2) {
      return jetTagValues[i1] > jetTagValues[i2];
    });

  // consider only up to m_MaxJetToSum indexes if the vector contains more element
  if(jetTagSortedIndices.size() > m_MaxJetToSum)
    jetTagSortedIndices.resize(m_MaxJetToSum);

  // sum jet tags and possibly take mean value
  float sumJetTag = 0;
  for (auto const & idx : jetTagSortedIndices) {
    sumJetTag += jetTagValues[idx];
  }

  if (m_UseMeanValue and not jetTagSortedIndices.empty()) {
    sumJetTag /= jetTagSortedIndices.size();
  }

  // build output collection
  bool accept = false;
  if (jetTagSortedIndices.size() >= m_MinJetToSum and 
      sumJetTag >= m_MinTag and 
      sumJetTag <= m_MaxTag) {
    accept = true;     
    for (auto const idx : jetTagSortedIndices) {
      filterproduct.addObject(m_TriggerType, jetRefCollection[idx]);
    }
  }

  LogDebug("HLTSumJetTag") << "Filter Result = " << accept << " (SumJetTag = " << sumJetTag
			   << ") [UseMeanValue = " << m_UseMeanValue << "]";
  return accept;
}

template <typename T>
float HLTSumJetTag<T>::findTagValueByMinDeltaR2(const T & jet,
                                                const reco::JetTagCollection & jetTags,
                                                float maxDeltaR2) const{
  float ret = -1000;
  for (const auto & jetTag : jetTags) {
    float tmpDR2 = reco::deltaR2(jet, *(jetTag.first));
    if (tmpDR2 < maxDeltaR2) {
      maxDeltaR2 = tmpDR2;
      ret = jetTag.second;
    }
  }

  return ret;
}
