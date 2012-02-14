/** \class HLTJetCollectionsFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include <typeinfo>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsFilter.h"

//
// constructors and destructor
//
template <typename jetType>
HLTJetCollectionsFilter<jetType>::HLTJetCollectionsFilter(const edm::ParameterSet& iConfig):
  HLTFilter(iConfig),
  inputTag_(iConfig.getParameter< edm::InputTag > ("inputTag")),
  originalTag_(iConfig.getParameter< edm::InputTag > ("originalTag")),
  minJetPt_(iConfig.getParameter<double> ("MinJetPt")),
  maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
  minNJets_(iConfig.getParameter<unsigned int> ("MinNJets")),
  triggerType_(iConfig.getParameter<int> ("triggerType"))
{
}

template <typename jetType>
HLTJetCollectionsFilter<jetType>::~HLTJetCollectionsFilter(){}

template <typename jetType>
void
HLTJetCollectionsFilter<jetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<edm::InputTag>("originalTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<double>("MinJetPt",30.0);
  desc.add<double>("MaxAbsJetEta",2.6);
  desc.add<unsigned int>("MinNJets",1);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTJetCollectionsFilter<jetType>).name()),desc);
}

// ------------ method called to produce the data  ------------
template <typename jetType>
bool
HLTJetCollectionsFilter<jetType>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  typedef vector<jetType> JetCollection;
  typedef edm::RefVector<JetCollection> JetRefVector;
  typedef edm::Ref<JetCollection> JetRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(originalTag_);

  Handle < JetCollectionVector > theJetCollectionsHandle;
  iEvent.getByLabel(inputTag_, theJetCollectionsHandle);
  const JetCollectionVector & theJetCollections = *theJetCollectionsHandle;

  // filter decision
  bool accept(false);
  std::set<JetRef> goodJetRefs;

  for (unsigned int collection = 0; collection < theJetCollections.size(); ++collection) {
    unsigned int numberOfGoodJets(0);
    const JetRefVector & refVector = theJetCollections[collection];

    typename JetRefVector::const_iterator jet(refVector.begin());
    for (; jet != refVector.end(); jet++) {
      JetRef jetRef(*jet);
      if (jetRef->pt() >= minJetPt_ && std::abs(jetRef->eta()) <= maxAbsJetEta_){
        numberOfGoodJets++;
        goodJetRefs.insert(jetRef);
      }
    }

    if (numberOfGoodJets >= minNJets_) {
      accept = true;
      // keep looping through collections to save all possible jets 
    }
  }

  // fill the filter object
  for (typename std::set<JetRef>::const_iterator ref = goodJetRefs.begin(); ref != goodJetRefs.end(); ++ref) {
    filterproduct.addObject(triggerType_, *ref);
  }

  return accept;
}
