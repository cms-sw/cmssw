#ifndef __PhysicsTools_PatAlgos_ModifiedObjectProducer_h__
#define __PhysicsTools_PatAlgos_ModifiedObjectProducer_h__

#include "PhysicsTools/PatAlgos/interface/ObjectModifier.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include <memory>

//templates to allow the filling of the reference to the original object if the object supports it
namespace{  
  template<typename T,typename Dummy = decltype(&T::addParentRef)>
    constexpr auto tracksParents(int){return true;}
  template<typename T>
    constexpr auto tracksParents(long){return false;}
  template<typename T1,typename T2,typename std::enable_if<tracksParents<T1>(0),int >::type = 0>
    void addParentRef(T1& obj,const edm::Ptr<T2>& ref){obj.addParentRef(ref);}
  template<typename T1,typename T2,typename std::enable_if<!tracksParents<T1>(0),int >::type = 0>
    void addParentRef(T1& obj,const edm::Ptr<T2>& ref){}

}

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
    ~ModifiedObjectProducer() override {}

    void beginLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup& evs) final {
      modifier_->setEventContent(evs);
    }
    
    void produce(edm::Event& evt,const edm::EventSetup& evs) final {
      edm::Handle<edm::View<T> > input;
      auto output = std::make_unique<Collection>();

      evt.getByToken(src_,input);
      output->reserve(input->size());
      modifier_->setEvent(evt);
      for( auto& ptr : input->ptrs() ){
        output->push_back(*ptr);
        T& obj = output->back();
	if(tracksParents<T>(0)) addParentRef(obj,ptr);
        modifier_->modify(obj);
      }
      evt.put(std::move(output));
    }

  private:
    edm::EDGetTokenT<edm::View<T> > src_;
    std::unique_ptr<Modifier> modifier_;
  };
}

#endif
