#include "HLTSumJetTagWithMatching.h"

template <typename T>
HLTSumJetTagWithMatching<T>::HLTSumJetTagWithMatching(const edm::ParameterSet& config)
  : HLTFilter(config),
    m_Jets(config.getParameter<edm::InputTag>("Jets")),
    m_JetTags(config.getParameter<edm::InputTag>("JetTags")),
    m_JetsToken(consumes<std::vector<T>>(m_Jets)), 
    m_JetTagsToken(consumes<reco::JetTagCollection>(m_JetTags)),
    m_MinTag(config.getParameter<double>("MinTag")),
    m_MaxTag(config.getParameter<double>("MaxTag")),
    m_MinJetToSum(config.getParameter<int>("MinJetToSum")),
    m_MaxJetToSum(config.getParameter<int>("MaxJetToSum")),
    m_deltaR(config.getParameter<double>("deltaR")),
    m_UseMeanValue(config.getParameter<double>("UseMeanValue")),
    m_TriggerType(config.getParameter<int>("TriggerType"))
{
  edm::LogInfo("") << " (HLTSumJetTagWithMatching) trigger cuts: " << std::endl
		   << "\ttype of        jets used: " << m_Jets.encode() << std::endl
		   << "\ttype of tagged jets used: " << m_JetTags.encode() << std::endl
		   << "\tmin/max tag value: [" << m_MinTag << ".." << m_MaxTag << "]" << std::endl
		   << "\tmin/max number of jets to sum: [" << m_MinJetToSum << ".." << m_MaxJetToSum << "]" << std::endl
		   << "\tuse mean or sum of tag values: " <<m_UseMeanValue << std::endl
		   << "\tdeltaR for matching: "<<m_deltaR<<std::endl
		   << "\tTriggerType: " << m_TriggerType << std::endl;
}

template <typename T>
HLTSumJetTagWithMatching<T>::~HLTSumJetTagWithMatching() = default;

template <typename T>
void HLTSumJetTagWithMatching<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("Jets", edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("JetTags", edm::InputTag("hltJetTagCollection"));
  desc.add<double>("MinTag", 0.0);
  desc.add<double>("MaxTag", 999999.0);
  desc.add<int>("MinJetToSum", 1);
  desc.add<int>("MaxJetToSum", 99);
  desc.add<bool>("UseMeanValue", false);
  desc.add<double>("deltaR", 0.1);
  desc.add<int>("TriggerType", 0);
  descriptions.add(defaultModuleLabel<HLTSumJetTagWithMatching<T>>(), desc);
}

//
// member functions
//
template <typename T>
float HLTSumJetTagWithMatching<T>::findTag (const T & jet, const reco::JetTagCollection & jetTags, float minDR){
  float tmpTag = -1000;
  for(auto jetT = jetTags.begin(); jetT != jetTags.end(); ++jetT){
    float tmpDR = reco::deltaR(jet,*(jetT->first));
    if(tmpDR < minDR){
      minDR = tmpDR;
      tmpTag = jetT->second;
    }
  }
  return tmpTag;
}

// ------------ method called to produce the data  ------------
template <typename T>
bool HLTSumJetTagWithMatching<T>::hltFilter(edm::Event& event,const edm::EventSetup& setup,
					    trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  edm::Handle<TCollection> h_Jets;
  event.getByToken(m_JetsToken, h_Jets);
  if (saveTags()) filterproduct.addCollectionTag(m_Jets);
  
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
  std::sort(jetTag_values.begin(),jetTag_values.end(),std::greater<>());
  
  //// Select only good tags
  std::vector<float> jetTag_values_selected;
  unsigned int nJet = 0;
  for(auto const & jetTag : jetTag_values){
    if(nJet >= m_MaxJetToSum) break;
    ++nJet;
    jetTag_values_selected.push_back(jetTag);
  }

  /// produce the output jet collection  
  nJet = 0;
  std::vector<TRef> jetRefCollection;
  float sumJetTag = 0;
  for (auto const & jet : *h_Jets) {  
    TRef  jetRef = TRef(h_Jets, nJet);
    auto tagValue  = findTag(jet,*h_JetTags,m_deltaR);  // find the tag associated to the jet    
    LogTrace("") << "Jet " << nJet << " : Et = " << jet.et() << " , tag value = " << tagValue;
    nJet++;
    if(std::find(jetTag_values_selected.begin(),jetTag_values_selected.end(),tagValue) != jetTag_values_selected.end()){  // find the tag
      jetRefCollection.push_back(jetRef);
      sumJetTag += tagValue;
    }
  }

  if(m_UseMeanValue) sumJetTag /= jetTag_values_selected.size();
  
  // Accept value of the filter
  bool accept = true;
  if(jetRefCollection.size() < m_MinJetToSum)
    accept = false;
  else{    
    if(sumJetTag >= m_MinTag and sumJetTag <= m_MaxTag){
      accept = true;
      for(auto const & jetRef: jetRefCollection){
	filterproduct.addObject(m_TriggerType,jetRef);
      }
    } 
    else
      accept = false;
  }
  
  edm::LogInfo("") << " trigger accept ? = " << accept << " nTag/nJet = " << jetRefCollection.size() << "/" << nJet << std::endl;  
  return accept;
}

typedef HLTSumJetTagWithMatching<reco::CaloJet> HLTSumCaloJetTagWithMatching;
typedef HLTSumJetTagWithMatching<reco::PFJet> HLTSumPFJetTagWithMatching;
DEFINE_FWK_MODULE(HLTSumCaloJetTagWithMatching);
DEFINE_FWK_MODULE(HLTSumPFJetTagWithMatching);
