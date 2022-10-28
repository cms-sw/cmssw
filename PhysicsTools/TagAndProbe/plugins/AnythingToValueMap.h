#ifndef PhysicsTools_TagAndProbe_interface_AnythingToValueMap_h
#define PhysicsTools_TagAndProbe_interface_AnythingToValueMap_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace pat {
  namespace helper {

    template <class Adaptor,
              class Collection = typename Adaptor::Collection,
              typename value_type = typename Adaptor::value_type>
    class AnythingToValueMap : public edm::stream::EDProducer<> {
    public:
      typedef typename edm::ValueMap<value_type> Map;
      typedef typename Map::Filler MapFiller;
      explicit AnythingToValueMap(const edm::ParameterSet& iConfig)
          : failSilently_(iConfig.getUntrackedParameter<bool>("failSilently", false)),
            src_(consumes<Collection>(iConfig.getParameter<edm::InputTag>("src"))),
            adaptor_(iConfig, consumesCollector()) {
        produces<Map>(adaptor_.label());
      }
      ~AnythingToValueMap() override {}

      void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    private:
      bool failSilently_;
      edm::EDGetTokenT<Collection> src_;
      Adaptor adaptor_;
    };

    template <class Adaptor, class Collection, typename value_type>
    void AnythingToValueMap<Adaptor, Collection, value_type>::produce(edm::Event& iEvent,
                                                                      const edm::EventSetup& iSetup) {
      edm::Handle<Collection> handle;
      iEvent.getByToken(src_, handle);
      if (handle.failedToGet() && failSilently_)
        return;

      bool adaptorOk = adaptor_.init(iEvent);
      if ((!adaptorOk) && failSilently_)
        return;

      std::vector<value_type> ret;
      ret.reserve(handle->size());

      adaptor_.run(*handle, ret);

      auto map = std::make_unique<Map>();
      MapFiller filler(*map);
      filler.insert(handle, ret.begin(), ret.end());
      filler.fill();
      iEvent.put(std::move(map), adaptor_.label());
    }

    template <class Adaptor,
              class Collection = typename Adaptor::Collection,
              typename value_type = typename Adaptor::value_type>
    class ManyThingsToValueMaps : public edm::stream::EDProducer<> {
    public:
      typedef typename edm::ValueMap<value_type> Map;
      typedef typename Map::Filler MapFiller;
      explicit ManyThingsToValueMaps(const edm::ParameterSet& iConfig)
          : failSilently_(iConfig.getUntrackedParameter<bool>("failSilently", false)),
            src_(consumes<Collection>(iConfig.getParameter<edm::InputTag>("collection"))),
            inputs_(iConfig.getParameter<std::vector<edm::InputTag> >("associations")) {
        for (std::vector<edm::InputTag>::const_iterator it = inputs_.begin(), ed = inputs_.end(); it != ed; ++it) {
          adaptors_.push_back(Adaptor(*it, iConfig, consumesCollector()));
          produces<Map>(adaptors_.back().label());
        }
      }
      ~ManyThingsToValueMaps() override {}

      void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    private:
      bool failSilently_;
      edm::EDGetTokenT<Collection> src_;
      std::vector<edm::InputTag> inputs_;
      std::vector<Adaptor> adaptors_;
    };

    template <class Adaptor, class Collection, typename value_type>
    void ManyThingsToValueMaps<Adaptor, Collection, value_type>::produce(edm::Event& iEvent,
                                                                         const edm::EventSetup& iSetup) {
      edm::Handle<Collection> handle;
      iEvent.getByToken(src_, handle);
      if (handle.failedToGet() && failSilently_)
        return;

      std::vector<value_type> ret;
      ret.reserve(handle->size());

      for (typename std::vector<Adaptor>::iterator it = adaptors_.begin(), ed = adaptors_.end(); it != ed; ++it) {
        ret.clear();
        if (it->run(iEvent, *handle, ret)) {
          auto map = std::make_unique<Map>();
          MapFiller filler(*map);
          filler.insert(handle, ret.begin(), ret.end());
          filler.fill();
          iEvent.put(std::move(map), it->label());
        } else {
          if (!failSilently_)
            throw cms::Exception("ManyThingsToValueMaps") << "Error in adapter " << it->label() << "\n";
        }
      }
    }

  }  // namespace helper
}  // namespace pat

#endif
