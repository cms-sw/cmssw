#ifndef __PhysicsTools_PatAlgos_ModifiedObjectProducer_h__
#define __PhysicsTools_PatAlgos_ModifiedObjectProducer_h__

#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include <memory>

namespace pat {

  template <class T>
  class ModifiedObjectProducer : public edm::stream::EDProducer<> {
  public:
    typedef std::vector<T> Collection;
    typedef pat::ObjectModifier<T> Modifier;

    ModifiedObjectProducer(const edm::ParameterSet& conf) {
      //set our input source
      src_ = consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("src"));
      //setup modifier
      const edm::ParameterSet& mod_config = conf.getParameter<edm::ParameterSet>("modifierConfig");
      modifier_ = std::make_unique<Modifier>(mod_config, consumesCollector());
      //declare products
      produces<Collection>();
    }
    ~ModifiedObjectProducer() override {}

    void produce(edm::Event& evt, const edm::EventSetup& evs) final {
      modifier_->setEventContent(evs);

      auto output = std::make_unique<Collection>();

      auto input = evt.getHandle(src_);
      output->reserve(input->size());

      modifier_->setEvent(evt);

      for (auto const& itr : *input) {
        output->push_back(itr);
        T& obj = output->back();
        modifier_->modify(obj);
      }

      evt.put(std::move(output));
    }

  private:
    edm::EDGetTokenT<edm::View<T> > src_;
    std::unique_ptr<Modifier> modifier_;
  };
}  // namespace pat

#endif
