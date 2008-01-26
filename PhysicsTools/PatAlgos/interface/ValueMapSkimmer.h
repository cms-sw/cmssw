#ifndef PhysicsTools_PatAlgos_interface_ValueMapSkimmer_h
#define PhysicsTools_PatAlgos_interface_ValueMapSkimmer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "PhysicsTools/PatUtils/interface/RefHelper.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat { namespace helper {

  template<typename value_type, typename AssoContainer = typename ::edm::ValueMap<value_type>, typename KeyType = typename reco::Candidate > 
  class ValueMapSkimmer : public edm::EDProducer {
    public:
      typedef typename edm::ValueMap<value_type> OutputMap;
      typedef typename OutputMap::Filler         MapFiller;
      typedef typename edm::View<KeyType>        ViewType;
      typedef typename edm::RefToBase<KeyType>   RefType;
      typedef typename edm::ValueMap<RefType>    BackRefMap;

      explicit ValueMapSkimmer(const edm::ParameterSet & iConfig) :
            failSilently_(iConfig.getUntrackedParameter<bool>("failSilently", false)),
            collection_(iConfig.getParameter<edm::InputTag>("collection")),
            association_(iConfig.getParameter<edm::InputTag>("association")),
            backrefs_(iConfig.getParameter<edm::InputTag>("backrefs"))
            { 
                produces< OutputMap >();
            }
      ~ValueMapSkimmer() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;

    private:
      bool failSilently_;
      edm::InputTag collection_, association_, backrefs_;
  };

template<typename value_type, typename AssoContainer, typename KeyType> 
void ValueMapSkimmer<value_type,AssoContainer,KeyType>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle< BackRefMap > backrefs;  
    iEvent.getByLabel(backrefs_, backrefs);

    edm::Handle< ViewType > collection;
    iEvent.getByLabel(collection_, collection);

    edm::Handle< AssoContainer > association;
    // needed in 16X as getByLabel throws immediatly
    try {
    iEvent.getByLabel(association_, association);
    if (association.failedToGet() && failSilently_) return; 
    } catch (cms::Exception &e) { if (failSilently_) return; throw; }

    size_t size = collection->size();

    std::vector<value_type> ret;
    ret.reserve(size);

    ::pat::helper::RefHelper<KeyType> refhelper(*backrefs) ;

    for (size_t i = 0; i < size; ++i) {
        RefType newRef = collection->refAt(i);
        ret.push_back( refhelper.recursiveLookup(newRef, *association) );
    }

    std::auto_ptr<OutputMap> map(new OutputMap());
    MapFiller filler(*map);
    filler.insert(collection, ret.begin(), ret.end());
    filler.fill();
    iEvent.put(map);
}

} } // namespace;
    

#endif
