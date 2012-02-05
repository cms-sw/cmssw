#ifndef GetJetsFromHLTobject_h
#define GetJetsFromHLTobject_h

//
// Package:    GetJetsFromHLTobject
// Class:      GetJetsFromHLTobject
// 
/**\class GetJetsFromHLTobject 

 Description: 
   HLT algorithms produced trigger::TriggerFilterObjectWithRefs containing the jets etc.
   that caused the trigger to fire. This class gets these jets
   and stores references to them directly in the event in a RefVector.

 Implementation:
   Switched to templates to produce calojets and PFJets when they are taken as inputs (Jyothsna)
*/
//
// Original Author:  Ian Tomalin
//

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class declaration
//

template <typename T> 
class GetJetsFromHLTobject : public edm::EDProducer {
public:
  typedef T JetType;
  typedef std::vector<JetType> JetCollection;

  explicit GetJetsFromHLTobject(const edm::ParameterSet&);
  virtual ~GetJetsFromHLTobject();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
private:
  edm::InputTag m_jets;

};


//
// constructors and destructor
//
template <typename T>
GetJetsFromHLTobject<T>::GetJetsFromHLTobject(const edm::ParameterSet& iConfig) :
  m_jets( iConfig.getParameter<edm::InputTag>("jets") )
{
  produces<JetCollection>();
}

template <typename T>
GetJetsFromHLTobject<T>::~GetJetsFromHLTobject() {}

template<typename T>
void 
GetJetsFromHLTobject<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag> ("jets", edm::InputTag("triggerFilterObjectWithRefs"));
  descriptions.add("hltGetJetsFromHLTobject", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void
GetJetsFromHLTobject<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  typedef edm::Ref<JetCollection> JetRef;

  std::auto_ptr<JetCollection> jets( new JetCollection() );

  // get hold of collection of TriggerFilterObjectsWithRefs
  edm::Handle<trigger::TriggerFilterObjectWithRefs> hltObject;
  iEvent.getByLabel(m_jets, hltObject);

  std::vector<JetRef> jetrefs;
  hltObject->getObjects( trigger::TriggerBJet, jetrefs );
  for (size_t i = 0; i < jetrefs.size(); ++i) {
    jets->push_back(* jetrefs[i]);
  }

  iEvent.put(jets);
}

#endif // GetJetsFromHLTobject_h
