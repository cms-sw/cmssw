/** \class HLTJetCollectionsFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTJetCollectionsFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

typedef std::vector<edm::RefVector<std::vector<reco::CaloJet>,reco::CaloJet,edm::refhelper::FindUsingAdvance<std::vector<reco::CaloJet>,reco::CaloJet> > > JetCollectionVector;

//
// constructors and destructor
//
HLTJetCollectionsFilter::HLTJetCollectionsFilter(const edm::ParameterSet& iConfig):
  inputTag_(iConfig.getParameter< edm::InputTag > ("inputTag")),
  saveTags_(iConfig.getParameter<bool>("saveTags")),
  minJetPt_(iConfig.getParameter<double> ("MinJetPt")),
  maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
  minNJets_(iConfig.getParameter<unsigned int> ("MinNJets"))
{
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTJetCollectionsFilter::~HLTJetCollectionsFilter(){}

void
HLTJetCollectionsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("MinJetPt",30.0);
  desc.add<double>("MaxAbsJetEta",2.6);
  desc.add<unsigned int>("MinNJets",1);
  descriptions.add("hltJetCollectionsFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTJetCollectionsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr < trigger::TriggerFilterObjectWithRefs > filterobject(new trigger::TriggerFilterObjectWithRefs(path(), module()));
  if (saveTags_) filterobject->addCollectionTag(inputTag_);

  Handle < JetCollectionVector > theCaloJetCollectionsHandle;
  iEvent.getByLabel(inputTag_, theCaloJetCollectionsHandle);
  const JetCollectionVector & theCaloJetCollections = *theCaloJetCollectionsHandle;

  // filter decision
  bool accept(false);
  std::vector < Ref<CaloJetCollection> > goodJetRefs;

  for (unsigned int collection = 0; collection < theCaloJetCollections.size(); ++collection) {
    unsigned int numberOfGoodJets(0);
    const reco::CaloJetRefVector & refVector = theCaloJetCollections[collection];

    if (refVector.size() < minNJets_) continue;

    //empty the good jets collection
    goodJetRefs.clear();

    Ref < CaloJetCollection > ref;
    reco::CaloJetRefVector::const_iterator jet(refVector.begin());
    for (; jet != refVector.end(); jet++) {
      reco::CaloJetRef jetRef(*jet);      
      if (jetRef->pt() >= minJetPt_ && fabs(jetRef->eta()) <= maxAbsJetEta_) {
	numberOfGoodJets++;
	ref = Ref<CaloJetCollection> (refVector, distance(refVector.begin(), jet));
	goodJetRefs.push_back(ref);	
      }
    }
    if (numberOfGoodJets >= minNJets_) {
      accept = true;
      break;
    }
  }

  //fill the filter object
  for (unsigned int refIndex = 0; refIndex < goodJetRefs.size(); ++refIndex) {
    filterobject->addObject(TriggerJet, goodJetRefs.at(refIndex));
  }

  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;

}
