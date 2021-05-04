#ifndef PhysicsTools_UtilAlgos_interface_EDFilterValueMapWrapper_h
#define PhysicsTools_UtilAlgos_interface_EDFilterValueMapWrapper_h

/**
 This is derived from EDFilterValueMapWrapper but rather than filtering it just stores a valuemap with the result
*/

#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  template <class T, class C>
  class FilterValueMapWrapper : public edm::stream::EDProducer<> {
  public:
    /// some convenient typedefs. Recall that C is a container class.
    typename C::iterator iterator;
    typename C::const_iterator const_iterator;

    /// default contructor. Declares the output (type "C") and the filter (of type T, operates on C::value_type)
    FilterValueMapWrapper(const edm::ParameterSet& cfg) : src_(consumes<C>(cfg.getParameter<edm::InputTag>("src"))) {
      filter_ = std::shared_ptr<T>(new T(cfg.getParameter<edm::ParameterSet>("filterParams")));
      produces<edm::ValueMap<int>>();
    }
    /// default destructor
    ~FilterValueMapWrapper() override {}
    /// everything which has to be done during the event loop. NOTE: We can't use the eventSetup in FWLite so ignore it
    void produce(edm::Event& event, const edm::EventSetup& eventSetup) override {
      // create a collection of the objects to put into the event
      auto objsToPut = std::make_unique<C>();
      // get the handle to the objects in the event.
      edm::Handle<C> h_c;
      event.getByToken(src_, h_c);
      std::vector<int> bitOut;
      // loop through and add passing value_types to the output vector
      for (typename C::const_iterator ibegin = h_c->begin(), iend = h_c->end(), i = ibegin; i != iend; ++i) {
        bitOut.push_back((*filter_)(*i));
      }
      std::unique_ptr<edm::ValueMap<int>> o(new edm::ValueMap<int>());
      edm::ValueMap<int>::Filler filler(*o);
      filler.insert(h_c, bitOut.begin(), bitOut.end());
      filler.fill();
      event.put(std::move(o));
    }

  protected:
    /// InputTag of the input source
    edm::EDGetTokenT<C> src_;
    /// shared pointer to analysis class of type BasicAnalyzer
    std::shared_ptr<T> filter_;
  };

}  // namespace edm

#endif
