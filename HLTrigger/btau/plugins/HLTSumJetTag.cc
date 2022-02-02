#include "HLTSumJetTag.h"

template <typename T>
HLTSumJetTag<T>::HLTSumJetTag(const edm::ParameterSet& config)
    : HLTFilter(config),
      m_Jets(config.getParameter<edm::InputTag>("Jets")),
      m_JetTags(config.getParameter<edm::InputTag>("JetTags")),
      m_JetsToken(consumes<std::vector<T>>(m_Jets)),
      m_JetTagsToken(consumes<reco::JetTagCollection>(m_JetTags)),
      m_MinTag(config.getParameter<double>("MinTag")),
      m_MaxTag(config.getParameter<double>("MaxTag")),
      m_MinJetToSum(config.getParameter<int>("MinJetToSum")),
      m_MaxJetToSum(config.getParameter<int>("MaxJetToSum")),
      m_UseMeanValue(config.getParameter<bool>("UseMeanValue")),
      m_TriggerType(config.getParameter<int>("TriggerType")) {
  edm::LogInfo("") << " (HLTSumJetTag) trigger cuts: \n"
                   << " \ttype of jets used: " << m_Jets.encode() << " \n"
                   << " \ttype of tagged jets used: " << m_JetTags.encode() << " \n"
                   << " \tmin/max tag value: [" << m_MinTag << ".." << m_MaxTag << "]"
                   << " \n"
                   << " \tmin/max number of jets to sum: [" << m_MinJetToSum << ".." << m_MaxJetToSum << "]"
                   << " \n"
                   << " \tuse mean value of jet tags: " << m_UseMeanValue << " \n"
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
  desc.add<int>("MaxJetToSum", 99);
  desc.add<bool>("UseMeanValue", true);
  desc.add<int>("TriggerType", 0);
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
  using namespace std;
  using namespace edm;
  using namespace reco;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  edm::Handle<TCollection> h_Jets;
  event.getByToken(m_JetsToken, h_Jets);
  if (saveTags())
    filterproduct.addCollectionTag(m_Jets);

  edm::Handle<JetTagCollection> h_JetTags;
  event.getByToken(m_JetTagsToken, h_JetTags);

  // check if the product this one depends on is available
  auto const& handle = h_JetTags;
  auto const& dependent = handle->keyProduct();
  if (not dependent.isNull() and not dependent.hasCache()) {
    // only an empty AssociationVector can have a invalid dependent collection
    edm::Provenance const& dependent_provenance = event.getProvenance(dependent.id());
    if (dependent_provenance.branchDescription().dropped())
      // FIXME the error message should be made prettier
      throw edm::Exception(edm::errors::ProductNotFound)
          << "Product " << handle.provenance()->branchName() << " requires product "
          << dependent_provenance.branchName() << ", which has been dropped";
  }

  //// Loop on jet tags and store values, sorting them in decreasing order
  std::vector<float> jetTag_values;
  for (auto const& jet : *h_JetTags)
    jetTag_values.push_back(jet.second);
  std::sort(jetTag_values.begin(), jetTag_values.end(), std::greater<>());

  //// apply max-jet requirement --> store only tags up to m_MaxJetToSum
  std::vector<float> jetTag_values_selected;
  unsigned int nJet = 0;
  for (auto const& jetTag : jetTag_values) {
    if (nJet >= m_MaxJetToSum)
      break;
    nJet++;
    jetTag_values_selected.push_back(jetTag);
  }

  /// produce jet collection
  std::vector<TRef> jetRefCollection;
  TRef jetRef;
  float sumJetTag = 0;
  nJet = 0;
  for (auto const& jet : *h_JetTags) {
    jetRef = TRef(h_Jets, jet.first.key());
    LogTrace("") << "Jet " << nJet << " : Et = " << jet.first->et() << " , tag value = " << jet.second << "\n";
    nJet++;
    if (std::find(jetTag_values_selected.begin(), jetTag_values_selected.end(), jet.second) !=
        jetTag_values_selected.end()) {  // if found
      jetRefCollection.push_back(jetRef);
      sumJetTag += jet.second;
    }
  }

  if (m_UseMeanValue and not jetTag_values_selected.empty())
    sumJetTag /= jetTag_values_selected.size();

  // Accept value of the filter
  bool accept = true;
  if (jetRefCollection.size() < m_MinJetToSum)
    accept = false;
  else {
    if (sumJetTag >= m_MinTag and sumJetTag <= m_MaxTag) {
      accept = true;
      for (auto const& jetRef : jetRefCollection) {
        filterproduct.addObject(m_TriggerType, jetRef);
      }
    } else
      accept = false;
  }

  edm::LogInfo("") << " trigger accept ? = " << accept << " nTag/nJet = " << jetRefCollection.size() << "/" << nJet
                   << " \n";

  return accept;
}
