#ifndef PhysicsTools_PatAlgos_interface_AnythingToValueMap_h
#define PhysicsTools_PatAlgos_interface_AnythingToValueMap_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace pat { namespace helper {

  template<class Adaptor, class Collection = typename Adaptor::Collection, typename value_type = typename Adaptor::value_type> 
  class AnythingToValueMap : public edm::EDProducer {
    public:
      typedef typename edm::ValueMap<value_type> Map;
      typedef typename Map::Filler MapFiller;
      explicit AnythingToValueMap(const edm::ParameterSet & iConfig) :
            failSilently_(iConfig.getUntrackedParameter<bool>("failSilently", false)),
            src_(iConfig.getParameter<edm::InputTag>("src")),
            adaptor_(iConfig) { 
                produces< Map >(adaptor_.label());
            }
      ~AnythingToValueMap() { }

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;

    private:
      bool failSilently_;
      edm::InputTag src_;
      Adaptor adaptor_;
  };

template<class Adaptor, class Collection, typename value_type>
void AnythingToValueMap<Adaptor,Collection,value_type>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle<Collection> handle;
    // needed in 16X, where getByLabel throws immediately making failedToGet is useless
    try {
    iEvent.getByLabel(src_, handle);
    if (handle.failedToGet() && failSilently_) return;

    bool adaptorOk = adaptor_.init(iEvent); 
    if ((!adaptorOk) && failSilently_) return;

    } catch (cms::Exception &ex) { if (failSilently_) return; throw; }

    std::vector<value_type> ret;
    ret.reserve(handle->size());

    adaptor_.run(*handle, ret);

    std::auto_ptr<Map> map(new Map());
    MapFiller filler(*map);
    filler.insert(handle, ret.begin(), ret.end());
	filler.fill();
    iEvent.put(map, adaptor_.label());
}

} } // namespace;
    

#endif
