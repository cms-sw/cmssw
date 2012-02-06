#ifndef HLTCollectionProducer_h
#define HLTCollectionProducer_h

//
// Package:    HLTCollectionProducer
// Class:      HLTCollectionProducer
// 
/**\class HLTCollectionProducer 

 Extract objects from trigger::TriggerFilterObjectWithRefs and fill
 them into a new collection (Based on GetJetsFromHLTobject.h)

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

#include <typeinfo>
#include <string>
#include <vector>

//
// class declaration
//

template <typename T> 
class HLTCollectionProducer : public edm::EDProducer {

  public:
    explicit HLTCollectionProducer(const edm::ParameterSet&);
    virtual ~HLTCollectionProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event&, const edm::EventSetup&);
  
  private:
    edm::InputTag    hltObject_;
    std::vector<int> triggerTypes_;
};


//
// constructors and destructor
//
template <typename T>
HLTCollectionProducer<T>::HLTCollectionProducer(const edm::ParameterSet& iConfig) :
  hltObject_    ( iConfig.getParameter<edm::InputTag>("HLTObject") ),
  triggerTypes_ ( iConfig.getParameter<std::vector<int> >("TriggerTypes") )
{
  produces< std::vector<T> >();
}

template <typename T>
HLTCollectionProducer<T>::~HLTCollectionProducer() {}

template<typename T>
void 
HLTCollectionProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag> ("HLTObject", edm::InputTag("TriggerFilterObjectWithRefs"));
  std::vector<int> triggerTypes;
  desc.add<std::vector<int> > ("TriggerTypes",triggerTypes);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTCollectionProducer<T>).name()), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void
HLTCollectionProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr<std::vector<T> > collection ( new std::vector<T>() );

  // get hold of collection of TriggerFilterObjectsWithRefs
  edm::Handle<trigger::TriggerFilterObjectWithRefs> hltObject;
  iEvent.getByLabel(hltObject_, hltObject);
  std::vector<edm::Ref<std::vector<T> > > objectRefs;

  for (size_t t=0; t<triggerTypes_.size(); ++t) {
    objectRefs.clear();
    hltObject->getObjects( triggerTypes_[t], objectRefs );
    for (size_t i = 0; i < objectRefs.size(); ++i) {
      collection->push_back(*objectRefs[i]);
    }
  }
  
  iEvent.put(collection);

}

#endif // HLTCollectionProducer_h
