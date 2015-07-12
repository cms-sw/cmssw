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
  
  template<class T>
  class ModifiedObjectProducer : public edm::stream::EDProducer<> {
  public:
    typedef std::vector<T> Collection;
    typedef pat::ObjectModifier<T> Modifier;

    ModifiedObjectProducer( const edm::ParameterSet& conf ) {
      //set our input source
      src_ = consumes<edm::View<T> >(conf.getParameter<edm::InputTag>("src"));
      //setup modifier
      edm::ConsumesCollector sumes(consumesCollector());      
      const edm::ParameterSet& mod_config = 
        conf.getParameter<edm::ParameterSet>("modifierConfig");
      modifier_.reset( new Modifier(mod_config) );
      modifier_->setConsumes(sumes);
      //declare products
      produces<Collection>();
    }
    ~ModifiedObjectProducer() {}

    virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& evs) override final {
      modifier_->setEventContent(evs);
    }
    
    virtual void produce(edm::Event& evt,const edm::EventSetup& evs) override final {
      edm::Handle<edm::View<T> > input;
      std::auto_ptr<Collection> output(new Collection);

      evt.getByToken(src_,input);
      output->reserve(input->size());

      modifier_->setEvent(evt);

      for( auto itr = input->begin(); itr != input->end(); ++itr ) {
        output->push_back(*itr);
        T& obj = output->back();
        modifier_->modify(obj);
      }

      evt.put(output);
    }

  private:
    edm::EDGetTokenT<edm::View<T> > src_;
    std::unique_ptr<Modifier> modifier_;
  };
}

#endif
