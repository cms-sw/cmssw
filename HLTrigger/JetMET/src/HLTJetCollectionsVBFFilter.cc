/** \class HLTJetCollectionsVBFFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTJetCollectionsVBFFilter.h"

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
HLTJetCollectionsVBFFilter::HLTJetCollectionsVBFFilter(const edm::ParameterSet& iConfig):
   inputTag_(iConfig.getParameter< edm::InputTag > ("inputTag")),
   saveTags_(iConfig.getParameter<bool>("saveTags")),
   softJetPt_(iConfig.getParameter<double> ("SoftJetPt")),
   hardJetPt_(iConfig.getParameter<double> ("HardJetPt")),
   minDeltaEta_(iConfig.getParameter<double> ("MinDeltaEta")), 
   thirdJetPt_(iConfig.getParameter<double> ("ThirdJetPt")),
   maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
   maxAbsThirdJetEta_(iConfig.getParameter<double> ("MaxAbsThirdJetEta")),
   minNJets_(iConfig.getParameter<unsigned int> ("MinNJets"))
{
   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTJetCollectionsVBFFilter::~HLTJetCollectionsVBFFilter(){}

void
HLTJetCollectionsVBFFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("SoftJetPt",25.0);
  desc.add<double>("HardJetPt",35.0);
  desc.add<double>("MinDeltaEta",3.0);
  desc.add<double>("ThirdJetPt",20.0);
  desc.add<double>("MaxAbsJetEta",9999.);
  desc.add<double>("MaxAbsThirdJetEta",2.6);
  desc.add<unsigned int>("MinNJets",2);
  descriptions.add("hltJetCollectionsVBFFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTJetCollectionsVBFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputTag_);

  Handle<JetCollectionVector> theCaloJetCollectionsHandle;
  iEvent.getByLabel(inputTag_,theCaloJetCollectionsHandle);
  const JetCollectionVector & theCaloJetCollections = *theCaloJetCollectionsHandle;
  // filter decision
  bool accept(false);
  std::vector < Ref<CaloJetCollection> > goodJetRefs;
  
  for(unsigned int collection = 0; collection < theCaloJetCollections.size(); ++ collection) {
    
    const reco::CaloJetRefVector & refVector =  theCaloJetCollections[collection];
    if(refVector.size() < minNJets_) continue;

    // VBF decision
    bool thereAreVBFJets(false);
    // 3rd Jet check decision
    bool goodThirdJet(false);
    if ( minNJets_ < 3 ) goodThirdJet = true;
    
    //empty the good jets collection
    goodJetRefs.clear();
            
    Ref<CaloJetCollection> refOne;
    Ref<CaloJetCollection> refTwo;
    reco::CaloJetRefVector::const_iterator jetOne ( refVector.begin() );
    int firstJetIndex=100, secondJetIndex=100, thirdJetIndex=100;

    // Cycle to look for VBF jets 
    for (; jetOne != refVector.end(); jetOne++) {
      reco::CaloJetRef jetOneRef(*jetOne);
            
      if ( thereAreVBFJets ) break;
      if ( jetOneRef->pt() < hardJetPt_ ) break;
      if ( fabs(jetOneRef->eta()) > maxAbsJetEta_ ) continue;
      
      reco::CaloJetRefVector::const_iterator jetTwo = jetOne + 1;
      secondJetIndex = firstJetIndex; 
      for (; jetTwo != refVector.end(); jetTwo++) {
        reco::CaloJetRef jetTwoRef(*jetTwo);
      
        if ( jetTwoRef->pt() < softJetPt_ ) break;
        if ( fabs(jetTwoRef->eta()) > maxAbsJetEta_ ) continue;
        
        if ( fabs(jetTwoRef->eta() - jetOneRef->eta()) < minDeltaEta_ ) continue;
        
        thereAreVBFJets = true;
        refOne = Ref<CaloJetCollection> (refVector, distance(refVector.begin(), jetOne));
        goodJetRefs.push_back(refOne);
        refTwo = Ref<CaloJetCollection> (refVector, distance(refVector.begin(), jetTwo));
        goodJetRefs.push_back(refTwo);
        
        firstJetIndex = (int) (jetOne - refVector.begin());
        secondJetIndex= (int) (jetTwo - refVector.begin());
        
        break;
        
      }
    }// Close looop on VBF
        
    // Look for a third jet, if you've found the previous 2
    if ( minNJets_ > 2 && thereAreVBFJets ) {
      Ref<CaloJetCollection> refThree;
      reco::CaloJetRefVector::const_iterator jetThree ( refVector.begin() );
      for (; jetThree != refVector.end(); jetThree++) {
        thirdJetIndex = (int) (jetThree - refVector.begin());

        reco::CaloJetRef jetThreeRef(*jetThree);
          
        if ( thirdJetIndex == firstJetIndex || thirdJetIndex == secondJetIndex ) continue;
      
        if (jetThreeRef->pt() >= thirdJetPt_ && fabs(jetThreeRef->eta()) <= maxAbsThirdJetEta_) {
          goodThirdJet = true;
          refThree = Ref<CaloJetCollection> (refVector, distance(refVector.begin(), jetThree));
          goodJetRefs.push_back(refThree);
          break;
        }
      }
    }

    if(thereAreVBFJets && goodThirdJet){
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
