/** \class HLTJetTagWithMatching
 *
 *  This class is an HLTFilter (a spcialized EDFilter) implementing
 *  tagged multi-jet trigger for b and tau.
 *  It should be run after the normal multi-jet trigger.
 *  It is like HLTJetTag, but it loop on jets collection instead of jetTag collection.
 *
 */

#include <vector>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "HLTJetTagWithMatching.h"


//
// constructors and destructor
//

template<typename T>
HLTJetTagWithMatching<T>::HLTJetTagWithMatching(const edm::ParameterSet & config) : HLTFilter(config),
  m_Jets   (config.getParameter<edm::InputTag>("Jets") ),
  m_JetTags(config.getParameter<edm::InputTag>("JetTags") ),
  m_MinTag (config.getParameter<double>        ("MinTag") ),
  m_MaxTag (config.getParameter<double>        ("MaxTag") ),
  m_MinJets(config.getParameter<int>           ("MinJets") ),
  m_TriggerType(config.getParameter<int>       ("TriggerType") ),
  m_deltaR (config.getParameter<double>           ("deltaR") )
{
  m_JetsToken = consumes<std::vector<T> >(m_Jets),
  m_JetTagsToken = consumes<reco::JetTagCollection>(m_JetTags),

  edm::LogInfo("") << " (HLTJetTagWithMatching) trigger cuts: " << std::endl
                   << "\ttype of        jets used: " << m_Jets.encode() << std::endl
                   << "\ttype of tagged jets used: " << m_JetTags.encode() << std::endl
                   << "\tmin/max tag value: [" << m_MinTag << ".." << m_MaxTag << "]" << std::endl
                   << "\tmin no. tagged jets: " << m_MinJets
		   << "\tTriggerType: " << m_TriggerType << std::endl;
}

template<typename T>
HLTJetTagWithMatching<T>::~HLTJetTagWithMatching()
{
}

template<typename T> float HLTJetTagWithMatching<T>::findCSV(const typename std::vector<T>::const_iterator & jet, const reco::JetTagCollection  & jetTags, float minDr){
         float tmpCSV = -20 ;
         for (reco::JetTagCollection::const_iterator jetb = jetTags.begin(); (jetb!=jetTags.end()); ++jetb) {
         float tmpDr = reco::deltaR(*jet,*(jetb->first));
 
         if (tmpDr < minDr) {
                 minDr = tmpDr ;
                 tmpCSV= jetb->second;
                 }
 
         }
         return tmpCSV;
 
 }
 
 
template<typename T>
void
HLTJetTagWithMatching<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("Jets",edm::InputTag("hltJetCollection"));
  desc.add<edm::InputTag>("JetTags",edm::InputTag("HLTJetTagWithMatchingCollection"));
  desc.add<double>("MinTag",2.0);
  desc.add<double>("MaxTag",999999.0);
  desc.add<int>("MinJets",1);
  desc.add<int>("TriggerType",0);
  desc.add<double>("deltaR",0.1);
  descriptions.add(defaultModuleLabel<HLTJetTagWithMatching<T>>(), desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTJetTagWithMatching<T>::hltFilter(edm::Event& event, const edm::EventSetup& setup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
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
  auto const & handle = h_JetTags;
  auto const & dependent = handle->keyProduct();
  if (not dependent.isNull() and not dependent.hasCache()) {
    // only an empty AssociationVector can have a invalid dependent collection
    edm::Provenance const & dependent_provenance = event.getProvenance(dependent.id());
    if (dependent_provenance.constBranchDescription().dropped())
      // FIXME the error message should be made prettier
      throw edm::Exception(edm::errors::ProductNotFound) << "Product " << handle.provenance()->branchName() << " requires product " << dependent_provenance.branchName() << ", which has been dropped";
  }

  TRef jetRef;

  // Look at all jets in decreasing order of Et.
  int nJet = 0;
  int nTag = 0;
  for (typename TCollection::const_iterator jet = h_Jets->begin(); jet != h_Jets->end(); ++jet) {
    jetRef = TRef(h_Jets,nJet);
    LogTrace("") << "Jet " << nJet
                 << " : Et = " << jet->et()
                 << " , tag value = " << findCSV(jet,*h_JetTags,m_deltaR);
    ++nJet;
    // Check if jet is tagged.
    if ( (m_MinTag <= findCSV(jet,*h_JetTags,m_deltaR)) and (findCSV(jet,*h_JetTags,m_deltaR) <= m_MaxTag) ) {
      ++nTag;

      // Store a reference to the jets which passed tagging cuts
      filterproduct.addObject(m_TriggerType,jetRef);
    }
  }

  // filter decision
  bool accept = (nTag >= m_MinJets);

  edm::LogInfo("") << " trigger accept ? = " << accept
                   << " nTag/nJet = " << nTag << "/" << nJet << std::endl;

  return accept;
}
